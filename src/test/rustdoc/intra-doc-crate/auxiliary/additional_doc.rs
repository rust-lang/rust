#![crate_name = "rand"]

pub trait RngCore {}
/// Rng extends [`RngCore`].
pub trait Rng: RngCore {}
