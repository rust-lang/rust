//~ ERROR values of the type `[u8; usize::MAX]` are too big for the target architecture
// Make sure the compiler does not ICE when trying to generate the debuginfo name of a type that
// causes a layout error. See https://github.com/rust-lang/rust/issues/94961.

//@ compile-flags:-C debuginfo=2
//@ build-fail

#![crate_type = "rlib"]

pub struct Foo<T>([T; usize::MAX]);

pub fn foo() -> usize {
    std::mem::size_of::<Foo<u8>>()
}
