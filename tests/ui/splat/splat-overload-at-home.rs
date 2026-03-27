//@ run-pass
// ignore-tidy-linelength
//! Test using `#[splat]` on some "overloading at home" example code.
//! <https://internals.rust-lang.org/t/pre-pre-rfc-splatting-for-named-arguments-and-function-overloading/24012>

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

    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //foo.method(());
    //foo.method((42i32,));

    // Generic tuple trait implementers work without explicit tuple type parameters.
    foo.method::<()>();
    foo.method();

    foo.method::<(i32,)>(42i32);
    foo.method::<(i32,)>(42);
    foo.method(42i32);
    foo.method(42);

    foo.method::<(i32, String)>(42i32, "asdf".to_owned());
    foo.method::<(i32, String)>(42, "asdf".to_owned());
    foo.method(42i32, "asdf".to_owned());
    foo.method(42, "asdf".to_owned());
}
