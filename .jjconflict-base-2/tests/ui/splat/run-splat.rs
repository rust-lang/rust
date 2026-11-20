//! Check that splat codegen works for simple cases.
//@ run-pass
#![feature(splat, tuple_trait)]
#![expect(incomplete_features)]

use std::marker::Tuple;

struct Foo;

trait MethodArgs: Tuple {
    fn call_method(self, this: &Foo) -> String;
}

impl Foo {
    fn method(&self, #[rustc_splat] args: impl MethodArgs) -> String {
        args.call_method(self)
    }
}

impl MethodArgs for (i32, String) {
    fn call_method(self, _this: &Foo) -> String {
        format!("{}-{}", self.0, self.1)
    }
}

impl MethodArgs for (f64,) {
    fn call_method(self, _this: &Foo) -> String {
        format!("{}", self.0)
    }
}

fn main() {
    let foo = Foo;
    assert_eq!(foo.method(42, "hello splat".to_string()), "42-hello splat");
    assert_eq!(foo.method(3.1), "3.1");
}
