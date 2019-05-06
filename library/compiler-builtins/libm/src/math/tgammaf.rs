use super::{tgamma};

pub fn tgammaf(x: f32) -> f32 {
    tgamma(x as f64) as f32
}
