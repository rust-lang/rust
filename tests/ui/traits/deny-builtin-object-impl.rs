//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]

#[rustc_deny_explicit_impl]
trait NotImplYesObject {}

#[rustc_deny_explicit_impl]
#[rustc_dyn_incompatible_trait]
trait NotImplNotObject {}

#[rustc_dyn_incompatible_trait]
trait YesImplNotObject {}

#[rustc_dyn_incompatible_trait]
trait YesImplNotObject2 {}

impl NotImplYesObject for () {}
//~^ ERROR explicit impls for the `NotImplYesObject` trait are not permitted

impl YesImplNotObject2 for dyn YesImplNotObject2 {}
//~^ ERROR the trait `YesImplNotObject2` is not dyn compatible

fn test_not_impl_yes_object<T: NotImplYesObject + ?Sized>() {}

fn test_not_impl_not_object<T: NotImplNotObject + ?Sized>() {}

fn test_yes_impl_not_object<T: YesImplNotObject + ?Sized>() {}

fn test_yes_impl_not_object2<T: YesImplNotObject2 + ?Sized>() {}

fn main() {
    test_not_impl_yes_object::<dyn NotImplYesObject>();

    test_not_impl_not_object::<dyn NotImplNotObject>();
    //~^ ERROR the trait `NotImplNotObject` is not dyn compatible

    test_yes_impl_not_object::<dyn YesImplNotObject>();
    //~^ ERROR the trait `YesImplNotObject` is not dyn compatible

    test_yes_impl_not_object2::<dyn YesImplNotObject2>();
    //~^ ERROR the trait `YesImplNotObject2` is not dyn compatible
}
