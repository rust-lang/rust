//! Checks that a set of measurements looks like a linear function rather than
//! like a quadratic function. Algorithm:
//!
//! 1. Linearly scale input to be in [0; 1)
//! 2. Using linear regression, compute the best linear function approximating
//!    the input.
//! 3. Compute RMSE and  maximal absolute error.
//! 4. Check that errors are within tolerances and that the constant term is not
//!    too negative.
//!
//! Ideally, we should use a proper "model selection" to directly compare
//! quadratic and linear models, but that sounds rather complicated:
//!
//! > https://stats.stackexchange.com/questions/21844/selecting-best-model-based-on-linear-quadratic-and-cubic-fit-of-data
//!
//! We might get false positives on a VM, but never false negatives. So, if the
//! first round fails, we repeat the ordeal three more times and fail only if
//! every time there's a fault.
use stdx::format_to;

#[derive(Default)]
pub struct AssertLinear {
    rounds: Vec<Round>,
}

#[derive(Default)]
struct Round {
    samples: Vec<(f64, f64)>,
    plot: String,
    linear: bool,
}

impl AssertLinear {
    pub fn next_round(&mut self) -> bool {
        if let Some(round) = self.rounds.last_mut() {
            round.finish();
        }
        if self.rounds.iter().any(|it| it.linear) || self.rounds.len() == 4 {
            return false;
        }
        self.rounds.push(Round::default());
        true
    }

    pub fn sample(&mut self, x: f64, y: f64) {
        self.rounds.last_mut().unwrap().samples.push((x, y));
    }
}

impl Drop for AssertLinear {
    fn drop(&mut self) {
        assert!(!self.rounds.is_empty());
        if self.rounds.iter().all(|it| !it.linear) {
            for round in &self.rounds {
                eprintln!("\n{}", round.plot);
            }
            panic!("Doesn't look linear!");
        }
    }
}

impl Round {
    fn finish(&mut self) {
        let (mut xs, mut ys): (Vec<_>, Vec<_>) = self.samples.iter().copied().unzip();
        normalize(&mut xs);
        normalize(&mut ys);
        let xy = xs.iter().copied().zip(ys.iter().copied());

        // Linear regression: finding a and b to fit y = a + b*x.

        let mean_x = mean(&xs);
        let mean_y = mean(&ys);

        let b = {
            let mut num = 0.0;
            let mut denom = 0.0;
            for (x, y) in xy.clone() {
                num += (x - mean_x) * (y - mean_y);
                denom += (x - mean_x).powi(2);
            }
            num / denom
        };

        let a = mean_y - b * mean_x;

        self.plot = format!("y_pred = {a:.3} + {b:.3} * x\n\nx     y     y_pred\n");

        let mut se = 0.0;
        let mut max_error = 0.0f64;
        for (x, y) in xy {
            let y_pred = a + b * x;
            se += (y - y_pred).powi(2);
            max_error = max_error.max((y_pred - y).abs());

            format_to!(self.plot, "{:.3} {:.3} {:.3}\n", x, y, y_pred);
        }

        let rmse = (se / xs.len() as f64).sqrt();
        format_to!(self.plot, "\nrmse = {:.3} max error = {:.3}", rmse, max_error);

        self.linear = rmse < 0.05 && max_error < 0.1 && a > -0.1;

        fn normalize(xs: &mut [f64]) {
            let max = xs.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            xs.iter_mut().for_each(|it| *it /= max);
        }

        fn mean(xs: &[f64]) -> f64 {
            xs.iter().copied().sum::<f64>() / (xs.len() as f64)
        }
    }
}
