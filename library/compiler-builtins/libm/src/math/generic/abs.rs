use super::super::Float;

/// Absolute value.
pub fn abs<F: Float>(x: F) -> F {
    x.abs()
}
