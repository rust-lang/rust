#![feature(unboxed_closures)]
#![feature(type_alias_impl_trait)]

type FunType = impl Fn<()>;
//~^ ERROR could not find defining uses
static STATIC_FN: FunType = some_fn;
//~^ ERROR mismatched types

fn some_fn() {}

fn main() {
    let _: <FunType as FnOnce<()>>::Output = STATIC_FN();
}
