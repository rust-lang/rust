// aux-build:baz.rs
// compile-flags:--extern baz
// edition:2018

mod foo {
    pub type Bar = u32;
}

mod bazz {
    use foo::Bar; //~ ERROR unresolved import `foo`

    fn baz() {
        let x: Bar = 22;
    }
}

use foo::Bar;

use foobar::Baz; //~ ERROR unresolved import `foobar`

fn main() { }
