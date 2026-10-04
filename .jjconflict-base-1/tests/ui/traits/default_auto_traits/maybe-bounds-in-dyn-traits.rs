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
trait PointeeSized: ?Leak {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized + ?Leak {}

#[lang = "sized"]
trait Sized: MetaSized + ?Leak {}

#[lang = "copy"]
pub trait Copy: ?Leak {}
impl<'a, T: ?Sized> Copy for &'a T {}

#[lang = "legacy_receiver"]
trait Receiver: ?Leak {}
impl<T: ?Sized + ?Leak> Receiver for &T {}
impl<T: ?Sized + ?Leak> Receiver for &mut T {}

#[lang = "unsize"]
trait Unsize<T: ?Sized + ?Leak>: ?Leak {}

#[lang = "coerce_unsized"]
trait CoerceUnsized<T: ?Leak + ?Sized>: ?Leak {}
impl<'a, 'b: 'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> CoerceUnsized<&'a U> for &'b T {}
// Omit `T: ?Leak` and `U: ?Leak`.
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'b mut T {}

#[lang = "dispatch_from_dyn"]
trait DispatchFromDyn<T: ?Leak>: ?Leak {}
impl<'a, T: ?Sized + ?Leak + Unsize<U>, U: ?Sized + ?Leak> DispatchFromDyn<&'a U> for &'a T {}
// Omit `T: ?Leak` and `U: ?Leak`.
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a mut U> for &'a mut T {}

#[lang = "default_trait1"]
auto trait Leak {}

struct NonLeakS;
impl !Leak for NonLeakS {}
struct LeakS;

fn bounds_check() {
    trait LeakTr {}

    trait MaybeLeakTr: ?Leak {}

    impl MaybeLeakTr for NonLeakS {}

    impl LeakTr for LeakS {}
    impl MaybeLeakTr for LeakS {}

    let _: &dyn LeakTr = &NonLeakS;
    //~^ ERROR  the trait bound `NonLeakS: bounds_check::LeakTr` is not satisfied
    let _: &dyn LeakTr = &LeakS;

    let _: &(dyn LeakTr + ?Leak) = &NonLeakS;
    let _: &(dyn LeakTr + ?Leak) = &LeakS;

    let _: &dyn MaybeLeakTr = &NonLeakS;
    let _: &dyn MaybeLeakTr = &LeakS;
}

fn dyn_compat_check() {
    trait DynCompatCheck1: ?Leak {
        fn foo(&self) {}
    }

    trait DynCompatCheck2: ?Leak {
        fn mut_foo(&mut self) {}
    }

    impl DynCompatCheck1 for NonLeakS {}
    impl DynCompatCheck2 for NonLeakS {}

    let _: &(dyn DynCompatCheck1 + ?Leak) = &NonLeakS;
    // There is no `?Leak` bound on corresponding `DispatchFromDyn` impl.
    let _: &dyn DynCompatCheck2 = &NonLeakS;
    //~^ ERROR the trait `DynCompatCheck2` is not dyn compatible
}

fn args_check() {
    trait LeakTr {}

    // Ensure that we validate the generic args of relaxed bounds in trait object types.
    let _: dyn LeakTr + ?Leak<(), Undefined = ()>;
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
    //~| ERROR associated type `Undefined` not found for `Leak`
}

fn main() {}
