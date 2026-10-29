//@ run-rustfix

#![allow(unused_imports)]

use foo::{
    #[cfg(true)]
    //~^ ERROR attributes are not allowed inside imports
    bar,
    #[cfg(false)]
    //~^ ERROR attributes are not allowed inside imports
    baz,
};

// Make sure we handle reserved symbols (leading `::` is `sym::PathRoot`).
use ::foo::{
    #[cfg(false)]
    //~^ ERROR attributes are not allowed inside imports
    qux,
};

mod foo {
    pub(crate) mod bar {}
    pub(crate) mod baz {}
    pub(crate) mod qux {}
}

fn main() {}
