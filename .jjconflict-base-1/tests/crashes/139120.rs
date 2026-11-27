//@ known-bug: #139120



pub trait Foo {
    type Bar<'a>;
}

pub struct FooImpl {}

impl Foo for FooImpl {
    type Bar<'a> = ();
}

pub trait FooFn {
    fn bar(&self);
}

impl<T: Foo> FooFn for fn(T, T::Bar<'_>) {
    fn bar(&self) {}
}

fn foo<T: Foo>(f: fn(T, T::Bar<'_>)) {
    let _: &dyn FooFn = &f;
}

fn main() {
    foo(|_: FooImpl, _| {});
}
