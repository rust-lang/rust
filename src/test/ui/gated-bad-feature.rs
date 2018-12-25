#![feature(
    foo_bar_baz,
    foo(bar),
    foo = "baz"
)]
//~^^^ ERROR: malformed feature
//~^^^ ERROR: malformed feature

#![feature] //~ ERROR: malformed feature
#![feature = "foo"] //~ ERROR: malformed feature

#![feature(test_removed_feature)] //~ ERROR: feature has been removed

fn main() {}
