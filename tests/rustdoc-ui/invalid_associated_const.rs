#![feature(associated_const_equality)]

trait T {
    type A: S<C<X = 0i32> = 34>;
    //~^ ERROR associated type bindings are not allowed here
}

trait S {
    const C: i32;
}
