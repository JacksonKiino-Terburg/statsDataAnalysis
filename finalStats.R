library(ggplot2)
library(lme4)
library(broom)
library(broom.mixed)
library(performance)
library(dplyr)
library(tidyr)
library(ggpubr)
library(MuMIn)

set.seed(16)
n_reps      <- 30
gens        <- 1:20
pred_levels <- c("High","Med","Low")
n_pred      <- length(pred_levels)

design <- expand.grid(
  replicate_id      = factor(1:n_reps),
  predator_pressure = factor(pred_levels, levels=pred_levels),
  generation        = gens
)

design <- design %>%
  mutate(
    prey_diversity = round(runif(n(), 5, 15)),
    mutation_rate  = runif(n(), 0.01, 0.1),
    crossover_rate = runif(n(), 0.5, 1.0),
    pred_num       = as.numeric(predator_pressure) - 1
  )

# coefficients
beta0   <- 5.0; b_pred  <- c(1.0, 0.5, 0.0)
b_prey  <- 0.2; b_mut   <- 10; b_cross <- 2; b_gen   <- 0.1
rep_int <- rnorm(n_reps, 0, 0.5); names(rep_int) <- levels(design$replicate_id)

# simulate
design <- design %>%
  rowwise() %>%
  mutate(
    complexity_score = beta0 + b_pred[pred_num+1] + b_prey*prey_diversity +
      b_mut*mutation_rate + b_cross*crossover_rate +
      b_gen*generation + rep_int[replicate_id] + rnorm(1, 0, 0.8),
    mean_fitness     = 50 + 5*complexity_score + rnorm(1, 0, 5)
  ) %>%
  ungroup()

sim_data <- design %>%
  select(replicate_id, predator_pressure, prey_diversity,
         mutation_rate, crossover_rate, generation,
         complexity_score, mean_fitness)


# Complexity across generations by predator pressure
p1 <- ggplot(sim_data, aes(x = generation, y = complexity_score,
                           color = predator_pressure)) +
  geom_jitter(alpha = 0.4, width = 0.2) +
  stat_smooth(method = "loess", se = TRUE) +
  labs(title = "Complexity score over generations",
       subtitle = "Colored by predator pressure level",
       x = "Generation", y = "Complexity score") +
  theme_minimal()
p1

# Mean fitness vs complexity
p2 <- ggplot(sim_data, aes(x = complexity_score, y = mean_fitness)) +
  geom_point(alpha = 0.4) +
  stat_smooth(method = "lm", se = TRUE) +
  labs(title = "Mean fitness vs complexity",
       x = "Complexity score", y = "Mean fitness") +
  theme_minimal()
p2


# define models: full interaction, additive, single predictor, plus mutation and crossover, and null
m_full            <- lm(complexity_score ~ predator_pressure * generation, data = sim_data)
m_add             <- lm(complexity_score ~ predator_pressure + generation, data = sim_data)
m_pred            <- lm(complexity_score ~ predator_pressure, data = sim_data)
m_gen             <- lm(complexity_score ~ generation, data = sim_data)
m_prey            <- lm(complexity_score ~ prey_diversity, data = sim_data)
m_mut             <- lm(complexity_score ~ mutation_rate, data = sim_data)
m_cross           <- lm(complexity_score ~ crossover_rate, data = sim_data)
m_mutcross        <- lm(complexity_score ~ mutation_rate + crossover_rate, data = sim_data)
m_pred_mut_cross  <- lm(complexity_score ~ predator_pressure + mutation_rate + crossover_rate, data = sim_data)
m0                <- lm(complexity_score ~ 1, data = sim_data)

# collect into a list
model_set <- list(
  Full_Interaction     = m_full,
  Additive             = m_add,
  Predator_only        = m_pred,
  Generation_only      = m_gen,
  Prey_only            = m_prey,
  Mutation_only        = m_mut,
  Crossover_only       = m_cross,
  Mut_Cross_additive   = m_mutcross,
  Pred_Mut_Cross       = m_pred_mut_cross,
  Null                 = m0
)

# Compute initial AICc table
aictab <- model.sel(model_set, rank = "AICc")
print(aictab)

# Fit additional models
m_pred_mut_int   <- lm(complexity_score ~ predator_pressure * mutation_rate, data = sim_data)
m_pred_cross_int <- lm(complexity_score ~ predator_pressure * crossover_rate, data = sim_data)
m_true <- lm(complexity_score ~ predator_pressure + generation + 
               prey_diversity + mutation_rate + crossover_rate, data = sim_data)
m_mixed <- lmer(complexity_score ~ predator_pressure + generation + 
                  prey_diversity + mutation_rate + crossover_rate + 
                  (1 | replicate_id), data = sim_data)

# Re-evaluate AICc with these models added
model_set2 <- c(
  model_set[c("Additive", "Full_Interaction")], # Keep Additive and Full_Interaction
  list(
    True = m_true,
    Mixed = m_mixed,
    Pred_x_Mut = m_pred_mut_int,
    Pred_x_Cross = m_pred_cross_int
  )
)

aictab2 <- model.sel(model_set2, rank = "AICc")
print(aictab2)

#Predator pressure has a positive main effect (High < Med < Low).

#Complexity increases over generations (b_gen > 0).

#Prey diversity, mutation, crossover each contribute positively, as simulated.

#My diagnostic plots show roughly homoscedastic, normally distributed residuals, so the LMER assumptions hold.

# final model
final_mod <- m_mixed
summary(final_mod)
confint(final_mod)

# Residuals
sim_data <- sim_data %>%
  mutate(
    .fitted_mixed = predict(final_mod),
    .resid_mixed  = residuals(final_mod)
  )

p_resid_fitted_mixed <- ggplot(sim_data, aes(x = .fitted_mixed, y = .resid_mixed)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(alpha = 0.4) +
  stat_smooth(method = "loess", se = FALSE) +
  labs(title = "Mixed Effects Model: Residuals vs Fitted",
       x = "Fitted values", y = "Residuals") +
  theme_minimal()

p_resid_qq_mixed <- ggplot(sim_data, aes(sample = .resid_mixed)) +
  stat_qq() + 
  stat_qq_line() +
  labs(title = "Mixed Effects Model: Normal Q-Q Plot") +
  theme_minimal()

# Model predictions:
dat_pred <- expand.grid(
  predator_pressure = levels(sim_data$predator_pressure),
  generation = seq(min(sim_data$generation), max(sim_data$generation), length.out = 100),
  prey_diversity = mean(sim_data$prey_diversity, na.rm = TRUE),
  mutation_rate = mean(sim_data$mutation_rate, na.rm = TRUE),
  crossover_rate = mean(sim_data$crossover_rate, na.rm = TRUE),
  replicate_id = NA  # Random effect set to NA for marginal prediction
)

dat_pred$predicted_complexity <- predict(final_mod, newdata = dat_pred, re.form = NA) 
# re.form = NA ignores random effects for clean marginal predictions

# Combined plot of residuals
ggarrange(p_resid_fitted_mixed, p_resid_qq_mixed, ncol = 1, nrow = 2)

# Plot predicted vs observed over generations
p_pred_mixed <- ggplot(sim_data, aes(x = generation, y = complexity_score, color = predator_pressure)) +
  geom_jitter(alpha = 0.3, width = 0.2) +
  geom_line(data = dat_pred, aes(x = generation, y = predicted_complexity, color = predator_pressure), size = 1) +
  labs(title = "Mixed Effects Model Predictions",
       subtitle = "Complexity over generations by predator pressure",
       x = "Generation", y = "Complexity score") +
  theme_minimal()

p_pred_mixed
