//@ edition:2018

use st::cell::Cell; //~ ERROR cannot find module or crate `st`

mod bar {
    pub fn bar() { bar::baz(); } //~ ERROR cannot find module or crate `bar`

    fn baz() {}
}

use bas::bar; //~ ERROR unresolved import `bas`

struct Foo {
    bar: st::cell::Cell<bool> //~ ERROR cannot find module or crate `st`
}

fn main() {}
