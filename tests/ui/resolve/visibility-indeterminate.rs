//@ edition:2018

foo!(); //~ ERROR cannot find macro `foo`

pub(in ::bar) struct Baz {} //~ ERROR cannot determine resolution for the visibility

fn main() {}
