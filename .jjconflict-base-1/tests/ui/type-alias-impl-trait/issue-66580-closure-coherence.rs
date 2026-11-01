// Regression test for issue #66580
// Ensures that we don't try to determine whether a closure
// is foreign when it's the underlying type of an opaque type
//@ check-pass
#![feature(type_alias_impl_trait)]

type Closure = impl FnOnce();

#[define_opaque(Closure)]
fn closure() -> Closure {
    || {}
}

struct Wrap<T> {
    f: T,
}

impl Wrap<Closure> {}

impl<T> Wrap<T> {}

fn main() {}
