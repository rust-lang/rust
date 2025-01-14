use super::super::Float;

pub fn fdim<F: Float>(x: F, y: F) -> F {
    if x <= y { F::ZERO } else { x - y }
}
