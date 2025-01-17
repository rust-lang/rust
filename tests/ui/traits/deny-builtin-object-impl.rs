//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]

#[rustc_deny_explicit_impl]
trait NotImplYesObject {}

#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
trait NotImplNotObject {}

#[rustc_do_not_implement_via_object]
trait YesImplNotObject {}

#[rustc_do_not_implement_via_object]
trait YesImplNotObject2 {}

impl NotImplYesObject for () {}
//~^ ERROR explicit impls for the `NotImplYesObject` trait are not permitted

// If there is no automatic impl then we can add a manual impl:
impl YesImplNotObject2 for dyn YesImplNotObject2 {}

fn test_not_impl_yes_object<T: NotImplYesObject + ?Sized>() {}

fn test_not_impl_not_object<T: NotImplNotObject + ?Sized>() {}

fn test_yes_impl_not_object<T: YesImplNotObject + ?Sized>() {}

fn test_yes_impl_not_object2<T: YesImplNotObject2 + ?Sized>() {}

fn main() {
    test_not_impl_yes_object::<dyn NotImplYesObject>();

    test_not_impl_not_object::<dyn NotImplNotObject>();
    //~^ ERROR the trait bound `dyn NotImplNotObject: NotImplNotObject` is not satisfied

    test_yes_impl_not_object::<dyn YesImplNotObject>();
    //~^ ERROR the trait bound `dyn YesImplNotObject: YesImplNotObject` is not satisfied

    test_yes_impl_not_object2::<dyn YesImplNotObject2>();
}
