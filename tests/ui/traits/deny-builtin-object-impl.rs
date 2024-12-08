//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]

#[rustc_deny_explicit_impl(implement_via_object = true)]
trait YesObject {}

#[rustc_deny_explicit_impl(implement_via_object = false)]
trait NotObject {}

fn test_yes_object<T: YesObject + ?Sized>() {}

fn test_not_object<T: NotObject + ?Sized>() {}

fn main() {
    test_yes_object::<dyn YesObject>();
    test_not_object::<dyn NotObject>();
    //~^ ERROR the trait bound `dyn NotObject: NotObject` is not satisfied
}
