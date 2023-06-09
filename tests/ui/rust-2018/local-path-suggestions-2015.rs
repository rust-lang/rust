// aux-build:baz.rs
// compile-flags:--extern baz
// edition:2015

// This test exists to demonstrate the behaviour of the import suggestions
// from the `local-path-suggestions-2018.rs` test when not using the 2018 edition.

extern crate baz as aux_baz;

mod foo {
    pub type Bar = u32;
}

mod baz {
    use foo::Bar;

    fn baz() {
        let x: Bar = 22;
    }
}

use foo::Bar;

use foobar::Baz; //~ ERROR unresolved import `foobar`

fn main() { }
