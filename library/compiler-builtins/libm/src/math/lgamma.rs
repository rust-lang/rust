use super::lgamma_r;

pub fn lgamma(x: f64) -> f64 {
    lgamma_r(x).0
}
