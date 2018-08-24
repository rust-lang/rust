#![feature(rustc_attrs)]

pub use bar::*;
mod bar {
    pub use super::*;
}

pub use baz::*;
mod baz {
    pub use main as f;
}

#[rustc_error]
pub fn main() {} //~ ERROR compilation successful
