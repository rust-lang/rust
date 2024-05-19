#![warn(clippy::deprecated_clippy_cfg_attr)]
#![allow(clippy::non_minimal_cfg)]
#![cfg_attr(feature = "cargo-clippy", doc = "a")] //~ ERROR: `feature = "cargo-clippy"` was

#[cfg_attr(feature = "cargo-clippy", derive(Debug))] //~ ERROR: `feature = "cargo-clippy"` was
#[cfg_attr(not(feature = "cargo-clippy"), derive(Debug))] //~ ERROR: `feature = "cargo-clippy"` was
#[cfg(feature = "cargo-clippy")] //~ ERROR: `feature = "cargo-clippy"` was
#[cfg(not(feature = "cargo-clippy"))] //~ ERROR: `feature = "cargo-clippy"` was
#[cfg(any(feature = "cargo-clippy"))] //~ ERROR: `feature = "cargo-clippy"` was
#[cfg(all(feature = "cargo-clippy"))] //~ ERROR: `feature = "cargo-clippy"` was
pub struct Bar;

fn main() {}
