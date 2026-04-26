#![feature(const_trait_impl)]
#![feature(const_closures)]

// Regression test for https://github.com/rust-lang/rust/issues/153891

const trait Foo {
    fn test() -> impl [const] Fn();
}

impl<T: Foo> Foo for &mut T {
    const fn test() -> impl [const] Fn() {
        //~^ ERROR functions in trait impls cannot be declared const
        const move || {}
    }
}

fn main() {}
