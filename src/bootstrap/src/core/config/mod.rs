#[expect(clippy::module_inception)]
mod config;
pub mod flags;
#[cfg(test)]
mod tests;

pub use config::*;
