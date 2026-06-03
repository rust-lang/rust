//! Test error cases for `#[splat]` "overloading at home" example code.
//! Splatted calls that don't match any registered MethodArgs impl should fail.
#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

struct Foo;

trait MethodArgs: std::marker::Tuple {
    fn call_method(self, _this: &Foo);
}

impl MethodArgs for () {
    fn call_method(self, _this: &Foo) {}
}

impl MethodArgs for (i32,) {
    fn call_method(self, _this: &Foo) {}
}

impl MethodArgs for (i32, String) {
    fn call_method(self, _this: &Foo) {}
}

impl Foo {
    fn method<T: MethodArgs>(&self, #[splat] args: T) {
        args.call_method(self)
    }
}

fn main() {
    let foo = Foo;

    // No impl for (f32,) — wrong type
    foo.method(42f32);
    //~^ ERROR mismatched types

    // No impl for (i32,i32) - wrong type
    foo.method(42i32, 42i32);
    //~^ ERROR mismatched types
}
