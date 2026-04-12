#![feature(const_trait_impl)]
#![feature(const_closures)]

// Regression test for https://github.com/rust-lang/rust/issues/153891

trait Foo {
    fn test() -> impl [const] Fn();
    //~^ ERROR `[const]` is not allowed here
}

impl<T: Foo> Foo for &mut T {
    const fn test() -> impl [const] Fn() {
        //~^ ERROR functions in trait impls cannot be declared const
        const move || {}
    }
}

fn main() {}
