//@ dont-require-annotations: NOTE

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait Foo: Copy + ToString {}

impl<T: Copy + ToString> Foo for T {}

fn hide<T: Foo>(x: T) -> impl Foo {
    x
}

trait Leak: Sized {
    type T;
    fn leak(self) -> Self::T;
}
impl<T> Leak for T {
    default type T = ();
    default fn leak(self) -> Self::T { panic!() }
}
impl Leak for i32 {
    type T = i32;
    fn leak(self) -> i32 { self }
}

fn main() {
    let _: u32 = hide(0_u32);
    //~^ ERROR mismatched types
    //~| NOTE expected type `u32`
    //~| NOTE found opaque type `impl Foo`
    //~| NOTE expected `u32`, found opaque type

    let _: i32 = Leak::leak(hide(0_i32));
    //~^ ERROR mismatched types
    //~| NOTE expected type `i32`
    //~| NOTE found associated type `<impl Foo as Leak>::T`
    //~| NOTE expected `i32`, found associated type

    let mut x = (hide(0_u32), hide(0_i32));
    x = (x.1,
    //~^ ERROR mismatched types
    //~| NOTE expected `u32`, found `i32`
         x.0);
    //~^ ERROR mismatched types
    //~| NOTE expected `i32`, found `u32`
}
