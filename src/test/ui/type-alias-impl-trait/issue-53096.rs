// check-pass
#![feature(const_impl_trait, const_fn_fn_ptr_basics)]
#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> usize;
const fn bar() -> Foo { || 0usize }
const BAZR: Foo = bar();

fn main() {}
