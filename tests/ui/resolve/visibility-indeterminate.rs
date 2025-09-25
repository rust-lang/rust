//@ edition:2018

foo!(); //~ ERROR cannot find macro `foo` in this scope

pub(in ::bar) struct Baz {} //~ ERROR failed to resolve: could not find `bar` in the list of imported crates

fn main() {}
