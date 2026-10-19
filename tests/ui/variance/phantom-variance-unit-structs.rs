//@ run-pass
#![feature(phantom_variance_markers, dropck_eyepatch, const_default, const_trait_impl)]

use std::cell::UnsafeCell;
use std::marker::*;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::rc::Rc;

const COV: PhantomCovariant<str> = PhantomCovariant;
const CONTRA: PhantomContravariant<str> = PhantomContravariant;
const INV: PhantomInvariant<str> = PhantomInvariant;
const COV_LT: PhantomCovariantLifetime<'static> = PhantomCovariantLifetime;
const CONTRA_LT: PhantomContravariantLifetime<'static> = PhantomContravariantLifetime;
const INV_LT: PhantomInvariantLifetime<'static> = PhantomInvariantLifetime;

fn check<T: Copy + Default + Eq + Ord + std::hash::Hash + std::fmt::Debug>(value: T) {
    assert_eq!(size_of::<T>(), 0);
    assert_eq!(align_of::<T>(), 1);
    assert!(!std::mem::needs_drop::<T>());
    assert_eq!(value, T::default());
}

fn auto_traits<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}

fn covariant<'a>(_: &'a (), x: PhantomCovariant<&'static ()>) -> PhantomCovariant<&'a ()> {
    x
}
fn contravariant<'a>(x: PhantomContravariant<&'a ()>) -> PhantomContravariant<&'static ()> {
    x
}
fn covariant_lifetime<'a>(
    _: &'a (),
    x: PhantomCovariantLifetime<'static>,
) -> PhantomCovariantLifetime<'a> {
    x
}
fn contravariant_lifetime<'a>(
    x: PhantomContravariantLifetime<'a>,
) -> PhantomContravariantLifetime<'static> {
    x
}

struct Dropper<T>(T);
unsafe impl<#[may_dangle] T> Drop for Dropper<T> {
    fn drop(&mut self) {}
}
struct Borrowed<'a>(&'a String);
impl Drop for Borrowed<'_> {
    fn drop(&mut self) {
        assert!(self.0.is_empty());
    }
}
fn markers<T>(_: &T) -> (PhantomCovariant<T>, PhantomContravariant<T>, PhantomInvariant<T>) {
    (PhantomCovariant, PhantomContravariant, PhantomInvariant)
}

fn main() {
    check(COV);
    check(CONTRA);
    check(INV);
    check(COV_LT);
    check(CONTRA_LT);
    check(INV_LT);
    assert!(matches!(const { PhantomCovariant::<str>::default() }, COV));
    assert!(matches!(const { PhantomContravariant::<str>::default() }, CONTRA));
    assert!(matches!(const { PhantomInvariant::<str>::default() }, INV));
    assert!(matches!(const { PhantomCovariantLifetime::<'static>::default() }, COV_LT));
    assert!(matches!(const { PhantomContravariantLifetime::<'static>::default() }, CONTRA_LT));
    assert!(matches!(const { PhantomInvariantLifetime::<'static>::default() }, INV_LT));
    let PhantomCovariant = COV;
    let PhantomContravariant = CONTRA;
    let PhantomInvariant = INV;
    let PhantomCovariantLifetime = COV_LT;
    let PhantomContravariantLifetime = CONTRA_LT;
    let PhantomInvariantLifetime = INV_LT;

    type NotAuto = (Rc<()>, PhantomPinned, UnsafeCell<()>);
    auto_traits::<PhantomCovariant<NotAuto>>();
    auto_traits::<PhantomContravariant<NotAuto>>();
    auto_traits::<PhantomInvariant<NotAuto>>();
    auto_traits::<PhantomCovariantLifetime<'static>>();
    auto_traits::<PhantomContravariantLifetime<'static>>();
    auto_traits::<PhantomInvariantLifetime<'static>>();

    assert_eq!(format!("{COV:?}"), "PhantomCovariant<str>");
    assert_eq!(format!("{CONTRA:?}"), "PhantomContravariant<str>");
    assert_eq!(format!("{INV:?}"), "PhantomInvariant<str>");
    assert_eq!(format!("{COV_LT:?}"), "PhantomCovariantLifetime");
    assert_eq!(format!("{CONTRA_LT:?}"), "PhantomContravariantLifetime");
    assert_eq!(format!("{INV_LT:?}"), "PhantomInvariantLifetime");

    let local = ();
    check(covariant(&local, PhantomCovariant));
    check(contravariant(PhantomContravariant));
    check(covariant_lifetime(&local, COV_LT));
    check(contravariant_lifetime(CONTRA_LT));

    // The markers do not own T for drop checking, even inside a type with a destructor.
    let _dropper;
    {
        let local = String::new();
        _dropper = Dropper(markers(&Borrowed(&local)));
    }
}
