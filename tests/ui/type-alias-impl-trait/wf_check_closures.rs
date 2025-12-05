#![feature(type_alias_impl_trait)]

trait Bar {
    fn bar(&self);
}

type FooFn<B> = impl FnOnce();

#[define_opaque(FooFn)]
fn foo<B: Bar>(bar: B) -> FooFn<B> {
    move || bar.bar()
    //~^ ERROR the trait bound `B: Bar` is not satisfied
}

fn main() {
    let boom: FooFn<u32> = unsafe { core::mem::zeroed() };
    boom();
}
