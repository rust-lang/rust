#![feature(unboxed_closures)]
#![feature(min_type_alias_impl_trait)]

type FunType = impl Fn<()>;
//~^ could not find defining uses
static STATIC_FN: FunType = some_fn;
//~^ mismatched types

fn some_fn() {}

fn main() {
    let _: <FunType as FnOnce<()>>::Output = STATIC_FN();
}
