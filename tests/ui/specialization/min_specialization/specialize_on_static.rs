// Test that directly specializing on `'static` is not allowed.

#![feature(min_specialization)]

trait X {
    fn f();
}

impl<T> X for &'_ T {
    default fn f() {}
}

impl X for &'static u8 {
    //~^ ERROR cannot specialize on `'static` lifetime
    fn f() {}
}

fn main() {}
