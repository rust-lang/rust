//@ edition:2018

foo!(); //~ ERROR cannot find macro `foo` in this scope

pub(in ::bar) struct Baz {} //~ ERROR cannot determine resolution for the visibility

fn main() {}
