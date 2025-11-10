// Regression test for <https://github.com/rust-lang/rust/issues/102467>.
// It ensures that the expected error is displayed.

#![expect(incomplete_features)]
#![feature(associated_const_equality, min_generic_const_args)]

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
