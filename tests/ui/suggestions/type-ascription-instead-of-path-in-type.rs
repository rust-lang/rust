enum A {
    B,
}

fn main() {
    let _: Vec<A:B> = A::B;
    //~^ ERROR cannot find trait `B` in this scope
    //~| HELP you might have meant to write a path instead of an associated type bound
    //~| ERROR associated type bounds are unstable
    //~| HELP add `#![feature(associated_type_bounds)]` to the crate attributes to enable
    //~| ERROR struct takes at least 1 generic argument but 0 generic arguments were supplied
    //~| HELP add missing generic argument
    //~| ERROR associated type bindings are not allowed here
}
