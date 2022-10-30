// Regression test for #90400,
// taken from https://github.com/rust-lang/rust/issues/90400#issuecomment-954927836

#![feature(type_alias_impl_trait)]

trait Bar {
    fn bar(&self);
}

trait Baz {
    fn baz(&self);
}

trait Foo {
    type FooFn<B>: Baz;

    fn foo<B: Bar>(&self, bar: B) -> Self::FooFn<B>;
}

struct MyFoo;
impl Foo for MyFoo {
    type FooFn<B> = impl Baz;

    fn foo<B: Bar>(&self, bar: B) -> Self::FooFn<B> {
        MyBaz(bar) //~ ERROR: the trait bound `B: Bar` is not satisfied
    }
}

struct MyBaz<B: Bar>(B);
impl<B: Bar> Baz for MyBaz<B> {
    fn baz(&self) {}
}

fn main() {
    let boom: <MyFoo as Foo>::FooFn<u32> = unsafe { core::mem::zeroed() };
    boom.baz();
}
