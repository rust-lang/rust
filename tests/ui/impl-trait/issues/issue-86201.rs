#![feature(unboxed_closures)]
#![feature(type_alias_impl_trait)]

// check-pass

type FunType = impl Fn<()>;
#[defines(FunType)]
static STATIC_FN: FunType = some_fn;

fn some_fn() {}

fn main() {
    let _: <FunType as FnOnce<()>>::Output = STATIC_FN();
}
