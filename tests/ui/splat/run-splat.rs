//! Check that splat codegen works for simple cases.
//@ run-pass
//@ check-run-results
#![feature(splat, tuple_trait)]
#![expect(incomplete_features)]

use std::marker::Tuple;

struct Foo;

trait MethodArgs: Tuple {
    fn call_method(self, this: &Foo);
}

impl Foo {
    fn method(&self, #[splat] args: impl MethodArgs) {
        args.call_method(self)
    }
}

impl MethodArgs for (i32, String) {
    fn call_method(self, _this: &Foo) {
        dbg!(self.1, self.0);
    }
}

impl MethodArgs for (f64,) {
    fn call_method(self, _this: &Foo) {
        dbg!(self.0);
    }
}

fn main() {
    let foo = Foo;
    foo.method(42, "hello splat".to_string());
    foo.method(3.141);
}
