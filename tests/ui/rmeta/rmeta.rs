//@ no-prefer-dynamic
//@ compile-flags: --emit=metadata

// Check that building a metadata crate finds an error.

fn main() {
    let _ = Foo; //~ ERROR cannot find value `Foo` in this scope
}
