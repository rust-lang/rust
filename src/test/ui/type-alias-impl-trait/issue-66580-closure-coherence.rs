// Regression test for issue #66580
// Ensures that we don't try to determine whether a closure
// is foreign when it's the underlying type of an opaque type
// check-pass
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type Closure = impl FnOnce();

fn closure() -> Closure {
    || {}
}

struct Wrap<T> { f: T }

impl Wrap<Closure> {}

impl<T> Wrap<T> {}

fn main() {}
