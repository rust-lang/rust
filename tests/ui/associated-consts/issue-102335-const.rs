#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

trait T {
    type A: S<C<X = 0i32> = 34>;
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR associated item constraints are not allowed here
}

trait S {
    #[type_const]
    const C: i32;
}

fn main() {}
