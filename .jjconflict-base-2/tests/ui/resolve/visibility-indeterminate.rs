//@ edition:2018

foo!(); //~ ERROR cannot find macro `foo` in this scope

pub(in ::bar) struct Baz {} //~ ERROR cannot find `bar`

fn main() {}
