// edition:2018

#![feature(crate_visibility_modifier)]

mod bar {
    crate struct Foo;
}

fn main() {
    Foo;
    //~^ ERROR cannot find value `Foo` in this scope [E0425]
}
