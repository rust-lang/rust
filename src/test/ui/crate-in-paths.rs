#![feature(crate_visibility_modifier)]
#![feature(crate_in_paths)]

mod bar {
    crate struct Foo;
}

fn main() {
    Foo;
    //~^ ERROR cannot find value `Foo` in this scope [E0425]
}
