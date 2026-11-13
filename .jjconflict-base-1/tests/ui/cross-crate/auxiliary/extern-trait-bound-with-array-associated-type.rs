//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/48984
#![crate_type = "lib"]
#![crate_name = "extern_trait_bound_with_array_associated_type"]

pub trait Foo { type Item; }

pub trait Bar: Foo<Item=[u8;1]> {  }
