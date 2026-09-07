// The type of a const argument's anon const is never computed, only fed while the enclosing body
// is type-checked. Under the parallel front end, `check_crate` used to ask for the `type_of` of a
// body owner nested inside such an anon const before that had happened, computing and caching an
// error type that then conflicted with the type fed later on.
//
// Each case below reaches the anon const through a different kind of path or position, none of
// which is resolved before the enclosing body is type-checked. They all live in one body so that
// the parallel front end cannot interleave their diagnostics.

//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

struct Foo<const N: u8>;
struct Tuple<const N: u8>(u8);

trait Trait {
    fn assoc<const N: u8>();
}

impl Trait for Foo<0> {
    fn assoc<const N: u8>() {}
}

impl<const N: u8> Foo<N> {
    fn inherent<const M: u8>() {}
}

mod submodule {
    pub fn free<const N: u8>() {}
}

fn free<const N: u8>() {}

fn preceded<T, const N: u8>() {}

// The late-bound `'a` is not a generic parameter, leaving this function with fewer parameters
// than it declares. Elided late-bound lifetimes in the signature do the same.
fn late_bound<'a, const N: u8>(_: &'a ()) {}

// `impl Sized` is a synthetic parameter, with no argument of its own.
fn synthetic<const N: u8>(_: impl Sized) {}

fn cases() {
    free::<{ async || {} }>();
    //~^ ERROR mismatched types
    submodule::free::<{ async || {} }>();
    //~^ ERROR mismatched types
    preceded::<u8, { async || {} }>();
    //~^ ERROR mismatched types
    late_bound::<{ async || {} }>(&());
    //~^ ERROR mismatched types
    synthetic::<{ async || {} }>(());
    //~^ ERROR mismatched types
    <Foo<0> as Trait>::assoc::<{ async || {} }>();
    //~^ ERROR mismatched types
    Foo::<0>::inherent::<{ async || {} }>();
    //~^ ERROR mismatched types
    let _ = Tuple::<{ async || {} }>(0);
    //~^ ERROR mismatched types

    // Not a path expression at all: the const argument sits in a type annotation, and is only
    // lowered once typeck reaches this statement.
    let _: Foo<{ async || {} }>;
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "\{async closure@[^`]*\}" -> "{async closure@...}"
