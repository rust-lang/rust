// Make sure the compiler does not ICE when trying to generate the debuginfo name of a type that
// causes a layout error.
// This version of the test already ICE'd before the commit that introduce the ICE described in
// https://github.com/rust-lang/rust/issues/94961.

//@ compile-flags:-C debuginfo=2 --error-format=human
//@ build-fail

#![crate_type = "rlib"]

pub enum Foo<T> {
    Bar([T; usize::MAX]),
}

pub fn foo() -> usize {
    std::mem::size_of::<Foo<u8>>()
}

// FIXME(#140620): the error is reported on different lines on different targets
//~? RAW values of the type `[u8; usize::MAX]` are too big for the target architecture
