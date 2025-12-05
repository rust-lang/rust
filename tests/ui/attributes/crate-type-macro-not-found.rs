// Tests for the issue in #137589
#[crate_type = foo!()] //~ ERROR cannot find macro `foo` in this scope
//~| WARN crate-level attribute should be an inner attribute

macro_rules! foo {
    ($x:expr) => {"rlib"}
}

fn main() {}
