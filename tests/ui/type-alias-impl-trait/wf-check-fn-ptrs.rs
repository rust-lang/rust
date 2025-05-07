#![feature(type_alias_impl_trait)]

//@ build-pass

trait Bar {
    fn bar(&self);
}

type FooFn<B> = impl FnOnce(B);

#[define_opaque(FooFn)]
fn foo<B: Bar>() -> FooFn<B> {
    fn mop<B: Bar>(bar: B) {
        bar.bar()
    }
    mop as fn(B)
    // function pointers don't have any obligations on them,
    // thus the above compiles. It's obviously unsound to just
    // procure a `FooFn` from the ether without making sure that
    // the pointer is actually legal for all `B`
}

fn main() {
    let boom: FooFn<u32> = unsafe { core::mem::zeroed() };
    boom(42);
}
