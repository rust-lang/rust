//@ aux-build: cross-crate.rs
extern crate cross_crate;

use cross_crate::{Bar, foo};

fn main() {
    foo::<true>();
    //~^ ERROR: function takes 0 generic arguments but 1 generic argument was supplied
    <() as Bar<true>>::bar();
    //~^ ERROR: trait takes 0 generic arguments but 1 generic argument was supplied
}

const FOO: () = {
    foo::<false>();
    //~^ ERROR: function takes 0 generic arguments but 1 generic argument was supplied
    <() as Bar<false>>::bar();
    //~^ ERROR: trait takes 0 generic arguments but 1 generic argument was supplied
};
