//@ compile-flags: -Zexperimental-default-bounds

#![feature(
    auto_traits,
    lang_items,
    more_maybe_bounds,
    negative_impls,
    no_core, start,
    trait_upcasting,
    rustc_attrs
)]
#![allow(internal_features)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
pub unsafe trait Copy {}
unsafe impl<'a, T: ?Sized> Copy for &'a T {}

#[lang = "receiver"]
trait Receiver {}
impl<T: ?Sized + ?Leak> Receiver for &T {}

#[lang = "unsize"]
trait Unsize<T: ?Sized + ?Leak> {}

#[lang = "coerce_unsized"]
trait CoerceUnsized<T: ?Leak> {}
impl<'a, 'b: 'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> CoerceUnsized<&'a U> for &'b T {}

#[lang = "dispatch_from_dyn"]
trait DispatchFromDyn<T: ?Leak> {}
impl<'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> DispatchFromDyn<&'a U> for &'a T {}

#[lang = "default_trait1"]
auto trait Leak {}

#[lang = "default_trait2"]
auto trait SyncDrop {}

struct NonLeakS;
impl !Leak for NonLeakS {}
struct LeakS;

trait Trait {
    fn leak_foo(&self) {}
    fn maybe_leak_foo(&self) where Self: ?Leak {}
}

impl Trait for NonLeakS {}
impl Trait for LeakS {}

// add implicit supertraits
trait Trait2<T = Self> {}
impl<T> Trait2<T> for LeakS {}

fn test_trait_object() {
    let _: &dyn Trait = &NonLeakS;
    //~^ ERROR the trait bound `NonLeakS: Leak` is not satisfied
    let _: &dyn Trait = &LeakS;
    let _: &(dyn Trait + ?Leak) = &LeakS;
    let x: &(dyn Trait + ?Leak) = &NonLeakS;
    x.leak_foo();
    //~^ ERROR the trait bound `dyn Trait: Leak` is not satisfied
    x.maybe_leak_foo();
}

fn test_traits_upcasting() {
    let _: &dyn Trait2<()> = &LeakS;
    let _: &dyn Leak = &LeakS;
    let _: &dyn SyncDrop = &LeakS;
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize { 0 }
