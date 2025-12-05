#![feature(unboxed_closures)]
#![feature(type_alias_impl_trait)]

//@ check-pass

type FunType = impl Fn<()>;
#[define_opaque(FunType)]
fn foo() -> FunType {
    some_fn
}

fn some_fn() {}

fn main() {
    let _: <FunType as FnOnce<()>>::Output = foo()();
}
