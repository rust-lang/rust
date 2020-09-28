// check-pass
#![feature(const_fn, const_fn_fn_ptr_basics)]
#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> usize;
const fn bar() -> Foo { || 0usize }
const BAZR: Foo = bar();

fn main() {}
