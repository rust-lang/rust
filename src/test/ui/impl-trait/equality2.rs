#![feature(specialization)]

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
    //~| expected type `u32`
    //~| found type `impl Foo`
    //~| expected u32, found opaque type

    let _: i32 = Leak::leak(hide(0_i32));
    //~^ ERROR mismatched types
    //~| expected type `i32`
    //~| found type `<impl Foo as Leak>::T`
    //~| expected i32, found associated type

    let mut x = (hide(0_u32), hide(0_i32));
    x = (x.1,
    //~^ ERROR mismatched types
    //~| expected u32, found i32
         x.0);
    //~^ ERROR mismatched types
    //~| expected i32, found u32
}
