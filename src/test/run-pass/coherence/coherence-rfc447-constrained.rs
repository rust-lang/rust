// run-pass
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
// check that trait matching can handle impls whose types are only
// constrained by a projection.

trait IsU32 {}
impl IsU32 for u32 {}

trait Mirror { type Image: ?Sized; }
impl<T: ?Sized> Mirror for T { type Image = T; }

trait Bar {}
impl<U: Mirror, V: Mirror<Image=L>, L: Mirror<Image=U>> Bar for V
    where U::Image: IsU32 {}

trait Foo { fn name() -> &'static str; }
impl Foo for u64 { fn name() -> &'static str { "u64" } }
impl<T: Bar> Foo for T { fn name() -> &'static str { "Bar" }}

fn main() {
    assert_eq!(<u64 as Foo>::name(), "u64");
    assert_eq!(<u32 as Foo>::name(), "Bar");
}
