// Test that `fn foo::bar::{self}` only imports `bar` in the type namespace.

mod foo {
    pub fn f() { }
}
use foo::f::{self}; //~ ERROR unresolved import `foo::f`

mod bar {
    pub fn baz() {}
    pub mod baz {}
}
use bar::baz::{self};

fn main() {
    baz(); //~ ERROR expected function, found module `baz`
}
