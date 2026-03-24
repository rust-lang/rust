//@ dont-check-compiler-stderr
//@ dont-check-failure-status
//@ dont-require-annotations: ERROR
// FIXME(splat): ^change the actual types during typeck so MIR doesn't ICE.

// ignore-tidy-linelength
//! Test using `#[splat]` on some "overloading at home" example code.
//! <https://internals.rust-lang.org/t/pre-pre-rfc-splatting-for-named-arguments-and-function-overloading/24012>

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

struct Foo;

trait MethodArgs: std::marker::Tuple {
    fn call_method(self, this: &Foo);
}
impl MethodArgs for () {
    fn call_method(self, this: &Foo) {}
}
impl MethodArgs for (i32,) {
    fn call_method(self, this: &Foo) {}
}
impl MethodArgs for (i32, String) {
    fn call_method(self, this: &Foo) {}
}

impl Foo {
    // FIXME(splat): make this work with impl MethodArgs
    fn method<T: MethodArgs>(&self, #[splat] args: T) {
        args.call_method(self)
    }
}

fn main() {
    let foo = Foo;

    // FIXME(splat):
    // - generic tuple trait implementers should work without explicit tuple type parameters.
    // - actually modify the argument list during typeck, to avoid "broken MIR" errors.
    foo.method::<()>(); //~ ERROR broken MIR
    foo.method(); //~ ERROR broken MIR

    foo.method::<(i32,)>(42i32); //~ ERROR broken MIR
    foo.method::<(i32,)>(42); //~ ERROR broken MIR
    foo.method(42i32); //~ ERROR broken MIR
    foo.method(42); //~ ERROR broken MIR
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //foo.method((42i32,));

    foo.method::<(i32, String)>(42i32, "asdf".to_owned()); //~ ERROR broken MIR
    foo.method::<(i32, String)>(42, "asdf".to_owned());
    foo.method(42i32, "asdf".to_owned());
    foo.method(42, "asdf".to_owned());
}
