#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]

pub type Foo = impl Fn() -> usize;
#[defines(Foo)]
pub const fn bar() -> Foo {
    || 0usize
}
const BAZR: Foo = bar();

#[rustc_error]
fn main() {} //~ ERROR
