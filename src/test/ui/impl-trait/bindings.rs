#![feature(impl_trait_in_bindings)]

fn a<T: Clone>(x: T) {
    const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
}

fn b<T: Clone>(x: T) {
    let _ = move || {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    };
}

trait Foo<T: Clone> {
    fn a(x: T) {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    }
}

impl<T: Clone> Foo<T> for i32 {
    fn a(x: T) {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    }
}

fn main() { }
