// Test type checking of uses of associated types via sugary paths.

pub trait Foo {
    type A;

    fn dummy(&self) { }
}

impl Foo for i32 {
    type A = u32;
}

pub fn f1<T: Foo>(a: T, x: T::A) {}
pub fn f2<T: Foo>(a: T) -> T::A {
    panic!();
}

pub fn f1_int_int() {
    f1(2i32, 4i32);
    //~^ ERROR mismatched types
    //~| expected u32, found i32
}

pub fn f1_int_uint() {
    f1(2i32, 4u32);
}

pub fn f1_uint_uint() {
    f1(2u32, 4u32);
    //~^ ERROR `u32: Foo` is not satisfied
    //~| ERROR `u32: Foo` is not satisfied
}

pub fn f1_uint_int() {
    f1(2u32, 4i32);
    //~^ ERROR `u32: Foo` is not satisfied
    //~| ERROR `u32: Foo` is not satisfied
}

pub fn f2_int() {
    let _: i32 = f2(2i32);
    //~^ ERROR mismatched types
    //~| expected i32, found u32
}

pub fn main() { }
