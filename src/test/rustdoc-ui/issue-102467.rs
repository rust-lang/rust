// failure-status: 101
//~^ ERROR

#![feature(associated_const_equality)]
#![feature(no_core)]

#![no_core]
#![crate_type = "lib"]

trait T {
    type A: S<C<X = 0i32> = 34>;
    //~^ ERROR
}

trait S {
    const C: i32;
}
