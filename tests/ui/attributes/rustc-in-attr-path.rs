mod rustc {
    pub use std::prelude::v1::test;
}

#[crate::rustc::test]
//~^ ERROR: attributes starting with `rustc` are reserved for use by the `rustc` compiler
fn foo() {}

fn main() {}
