#![feature(type_alias_impl_trait)]

trait Bar {
    fn bar(&self);
}

type FooFn<B> = impl FnOnce(B);

#[define_opaque(FooFn)]
fn foo<B: Bar>() -> FooFn<B> {
    fn mop<B: Bar>(bar: B) {
        bar.bar()
    }
    mop // NOTE: no function pointer, but function zst item
    //~^ ERROR the trait bound `B: Bar` is not satisfied
}

fn main() {
    let boom: FooFn<u32> = unsafe { core::mem::zeroed() };
    boom(42);
}
