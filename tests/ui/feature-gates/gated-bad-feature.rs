#![feature(foo_bar_baz, foo(bar), foo = "baz", foo)]
//~^ ERROR malformed `feature`
//~| ERROR malformed `feature`
//~| ERROR unknown feature `foo`
//~| ERROR unknown feature `foo_bar_baz`
#![feature] //~ ERROR malformed `feature` attribute
#![feature = "foo"] //~ ERROR malformed `feature` attribute
#![feature(test_removed_feature)] //~ ERROR: feature has been removed

fn main() {}
