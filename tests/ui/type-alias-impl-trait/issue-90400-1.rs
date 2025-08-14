// Regression test for #90400,
// taken from https://github.com/rust-lang/rust/issues/90400#issuecomment-954927836

#![feature(impl_trait_in_assoc_type)]

trait Bar {
    fn bar(&self);
}

trait Foo {
    type FooFn<B>: FnOnce();

    fn foo<B: Bar>(&self, bar: B) -> Self::FooFn<B>;
}

struct MyFoo;

impl Foo for MyFoo {
    type FooFn<B> = impl FnOnce();

    fn foo<B: Bar>(&self, bar: B) -> Self::FooFn<B> {
        move || bar.bar() //~ ERROR: the trait bound `B: Bar` is not satisfied
    }
}

fn main() {
    let boom: <MyFoo as Foo>::FooFn<u32> = unsafe { core::mem::zeroed() };
    boom();
}
