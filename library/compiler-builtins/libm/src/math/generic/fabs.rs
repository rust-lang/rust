use super::super::Float;

/// Absolute value.
pub fn fabs<F: Float>(x: F) -> F {
    x.abs()
}
