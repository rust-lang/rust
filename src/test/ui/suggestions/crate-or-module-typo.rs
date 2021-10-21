// edition:2018

#![feature(decl_macro)]

macro a() {
    extern crate core as my_core;
    use my_cor::mem;
    mod a {
        pub fn bar() {}
    }
}

macro_rules! b {
    () => {
        mod b {
            pub fn bar() {}
        }
    }
}

mod foo {
    pub fn bar() { fooo::baz(); } //~ ERROR failed to resolve: use of undeclared crate or module `fooo`

    fn baz() {}
}

a!();

b!();

use my_cor::mem;

use my_core::mem;

use aa::bar; //~ ERROR unresolved import `aa`

use bb::bar; //~ ERROR unresolved import `bb`

use st::cell::Cell; //~ ERROR failed to resolve: use of undeclared crate or module `st`

use fooo::bar; //~ ERROR unresolved import `fooo`

struct Foo {
    bar: st::cell::Cell<bool> //~ ERROR failed to resolve: use of undeclared crate or module `st`
}

fn main() {}
