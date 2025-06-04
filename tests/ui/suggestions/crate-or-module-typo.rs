//@ edition:2018

use st::cell::Cell; //~ ERROR cannot find item `st`

mod bar {
    pub fn bar() { bar::baz(); } //~ ERROR cannot find item `bar`

    fn baz() {}
}

use bas::bar; //~ ERROR unresolved import `bas`

struct Foo {
    bar: st::cell::Cell<bool> //~ ERROR cannot find item `st`
}

fn main() {}
