//@ check-pass

mod foo {
    pub use crate::bar::*;
    pub use crate::main as f;
}

mod bar {
    pub use crate::foo::*;
}

pub use foo::*;
pub use baz::*;
mod baz {
    pub use super::*;
}

pub fn main() {}
