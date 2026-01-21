#![crate_type = foo!()]
//~^ ERROR cannot find macro `foo`
//~| WARN this was previously accepted

macro_rules! foo {
    () => {"rlib"};
}

fn main() {}
