// check-pass
#![feature(const_fn)]
#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> usize;
const fn bar() -> Foo { || 0usize }
const BAZR: Foo = bar();

fn main() {}
