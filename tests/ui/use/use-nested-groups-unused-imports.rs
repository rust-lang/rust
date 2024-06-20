#![allow(dead_code)]
#![deny(unused_imports)]

mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Bar();
        }
        pub mod foobar {}
    }

    pub struct Foo();
}

use foo::{Foo, bar::{baz::{}, foobar::*}, *};
    //~^ ERROR unused imports: `*`, `Foo`, `baz::{}`, and `foobar::*`
use foo::bar::baz::{*, *};
    //~^ ERROR unused import: `*`
use foo::{};
    //~^ ERROR unused import: `foo::{}`

fn main() {
    let _: Bar;
}
