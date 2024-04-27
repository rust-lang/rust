pub use std::panic;

#[macro_export]
macro_rules! panic { () => {} } //~ ERROR the name `panic` is defined multiple times

fn main() {}
