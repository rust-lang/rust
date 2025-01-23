//@ compile-flags: -Zexperimental-default-bounds

#![feature(
    auto_traits,
    lang_items,
    more_maybe_bounds,
    negative_impls,
    no_core,
    rustc_attrs
)]
#![allow(internal_features)]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "copy"]
pub trait Copy {}
impl<'a, T: ?Sized> Copy for &'a T {}

#[lang = "legacy_receiver"]
trait Receiver {}
impl<T: ?Sized + ?Leak> Receiver for &T {}

#[lang = "unsize"]
trait Unsize<T: ?Sized + ?Leak> {}

#[lang = "coerce_unsized"]
trait CoerceUnsized<T: ?Leak + ?Sized> {}
impl<'a, 'b: 'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> CoerceUnsized<&'a U> for &'b T {}

#[lang = "dispatch_from_dyn"]
trait DispatchFromDyn<T: ?Leak> {}
impl<'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> DispatchFromDyn<&'a U> for &'a T {}

#[lang = "default_trait1"]
auto trait Leak {}

struct NonLeakS;
impl !Leak for NonLeakS {}
struct LeakS;

trait Trait {
    fn leak_foo(&self) {}
    fn maybe_leak_foo(&self) where Self: ?Leak {}
}

impl Trait for NonLeakS {}
impl Trait for LeakS {}

fn main() {
    let _: &dyn Trait = &NonLeakS;
    //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
    let _: &dyn Trait = &LeakS;
    let _: &(dyn Trait + ?Leak) = &LeakS;
    let x: &(dyn Trait + ?Leak) = &NonLeakS;
    x.leak_foo();
    //~^ ERROR the trait bound `dyn Trait: Leak` is not satisfied
    x.maybe_leak_foo();
}
