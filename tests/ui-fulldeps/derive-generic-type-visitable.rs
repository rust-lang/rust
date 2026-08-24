//@ edition: 2024

#![crate_type = "rlib"]
#![feature(rustc_private)]

extern crate rustc_type_ir;
extern crate rustc_type_ir_macros;

use rustc_type_ir::{GenericTypeVisitable, Interner};
use rustc_type_ir_macros::GenericTypeVisitable;

#[derive(GenericTypeVisitable)]
struct DerivesGenericTypeVisitable;

#[derive(GenericTypeVisitable)]
struct Foo {
    one: u32,
    two: Vec<i32>,
    three: String,
}

#[derive(GenericTypeVisitable)]
enum Enum {
    One,
    Two(u32),
    Three { two: Vec<i32>, three: String },
}

#[derive(GenericTypeVisitable)]
struct Generic<T>([T; 5]);

#[derive(GenericTypeVisitable)]
struct GenericOverInterner<I: Interner>(I::Clause);

#[derive(GenericTypeVisitable)]
struct Recursive {
    #[generic_type_visitable(bounds())]
    rec: Vec<Self>,
    other: u32,
}

#[derive(GenericTypeVisitable)]
struct PartiallyRecursiveField<T> {
    #[generic_type_visitable(bounds(T: GenericTypeVisitable<__V>))]
    partially_rec: (Vec<Self>, T),
    other: u32,
}

#[derive(GenericTypeVisitable)]
struct MissingBound<T> {
    // This should fail, as `T: GenericTypeVisitable<__V>` wasn't specified
    #[generic_type_visitable(bounds())]
    //~^ ERROR: the trait bound `T: GenericTypeVisitable<__V>` is not satisfied
    partially_rec: (Vec<Self>, T),
    other: u32,
}
