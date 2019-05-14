#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

fn a<T: Clone>(x: T) {
    const foo: impl Clone = x;
    //~^ ERROR attempt to use a non-constant value in a constant
}

fn b<T: Clone>(x: T) {
    let _ = move || {
        const foo: impl Clone = x;
        //~^ ERROR attempt to use a non-constant value in a constant
    };
}

trait Foo<T: Clone> {
    fn a(x: T) {
        const foo: impl Clone = x;
        //~^ ERROR attempt to use a non-constant value in a constant
    }
}

impl<T: Clone> Foo<T> for i32 {
    fn a(x: T) {
        const foo: impl Clone = x;
        //~^ ERROR attempt to use a non-constant value in a constant
    }
}

fn main() { }
