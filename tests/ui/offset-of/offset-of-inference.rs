// Test that inference types in `offset_of!` don't ICE.

struct Foo<T> {
    x: T,
}

fn main() {
    let _ = core::mem::offset_of!(Foo<_>, x); //~ ERROR: type annotations needed
}
