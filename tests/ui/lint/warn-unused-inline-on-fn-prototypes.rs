#![deny(unused_attributes)]

trait Trait {
    #[inline] //~ ERROR attribute cannot be used on
    //~^ WARN previously accepted
    fn foo();
}

extern "C" {
    #[inline] //~ ERROR attribute cannot be used on
    //~^ WARN previously accepted
    fn foo();
}

fn main() {}
