#![feature(const_impl_trait, const_fn_fn_ptr_basics, rustc_attrs)]
#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> usize;
const fn bar() -> Foo {
    || 0usize
}
const BAZR: Foo = bar();

#[rustc_error]
fn main() {} //~ ERROR
