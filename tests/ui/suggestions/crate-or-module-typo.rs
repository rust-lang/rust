//@ edition:2018

use st::cell::Cell; //~ ERROR failed to resolve: use of undeclared crate or module `st`

mod bar {
    pub fn bar() { bar::baz(); } //~ ERROR failed to resolve: function `bar` is not a crate or module

    fn baz() {}
}

use bas::bar; //~ ERROR unresolved import `bas`

struct Foo {
    bar: st::cell::Cell<bool> //~ ERROR failed to resolve: use of undeclared crate or module `st`
}

fn main() {}
