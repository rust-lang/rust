//@ check-pass

pub use bar::*;
mod bar {
    pub use super::*;
}

pub use baz::*;
mod baz {
    pub use crate::main as f;
}

pub fn main() {}
