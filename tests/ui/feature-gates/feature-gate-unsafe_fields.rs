//@ compile-flags: --crate-type=lib
//@ revisions: with_gate without_gate
//@ [with_gate] check-pass

#![cfg_attr(with_gate, feature(unsafe_fields))] //[with_gate]~ WARNING

#[cfg(any())]
struct Foo {
    unsafe field: (), //[without_gate]~ ERROR
}

// This should not parse as an unsafe field definition.
struct FooTuple(unsafe fn());

#[cfg(any())]
enum Bar {
    Variant { unsafe field: () }, //[without_gate]~ ERROR
    // This should not parse as an unsafe field definition.
    VariantTuple(unsafe fn()),
}

#[cfg(any())]
union Baz {
    unsafe field: (), //[without_gate]~ ERROR
}
