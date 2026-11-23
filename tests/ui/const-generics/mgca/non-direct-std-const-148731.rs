//! Regression test for <https://github.com/rust-lang/rust/issues/148731>.
#![feature(gca_min_const_items, gca_macroless_args)]
#![crate_type = "lib"]

pub trait Foo {}
impl Foo for [u8; std::path::MAIN_SEPARATOR] {}
//~^ ERROR use of `const` in the type system not marked as direct
