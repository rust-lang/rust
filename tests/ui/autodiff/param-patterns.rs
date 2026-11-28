//! Regression test for <https://github.com/rust-lang/rust/issues/150328>.
//! A parameter that is not a plain name, like `_` or `(a, b)`, used to ICE in expansion. The
//! generated function only passes its arguments on, so it binds each of them to a plain name.

//@ revisions: check enzyme
//@[check] check-pass
//@[enzyme] needs-enzyme
//@[enzyme] compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
//@[enzyme] no-prefer-dynamic
//@[enzyme] build-pass

#![feature(autodiff)]

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_reverse(d_wild, Active, Active, Active)]
fn wild(_: f64, y: f64) -> f64 {
    y * y
}

#[autodiff_reverse(d_tuple, Duplicated, Active)]
fn tuple(&(a, b): &(f64, f64)) -> f64 {
    a * b
}

#[autodiff_reverse(d_deref, Duplicated, Active)]
fn deref(&x: &f64) -> f64 {
    x * x
}

// Enzyme does not handle tuples passed by value yet, so `binding` and `dual` are in `check` only.
#[cfg(check)]
#[autodiff_reverse(d_binding, Const, Active)]
fn binding(whole @ (inner, _): (f64, f64)) -> f64 {
    whole.0 * inner
}

// `mut` and `ref` stay on the original function only.
#[autodiff_reverse(d_mut_ref, Active, Active, Active)]
fn mut_ref(mut x: f64, ref y: f64) -> f64 {
    x *= *y;
    x
}

#[cfg(check)]
#[autodiff_forward(d_dual, Dual, Const, Dual)]
fn dual(_: f64, (a, _): (f64, f64)) -> f64 {
    a
}

#[autodiff_forward(d_dual_ref, Dual, Const, Dual)]
fn dual_ref(_: f64, &(a, _): &(f64, f64)) -> f64 {
    a
}

#[autodiff_forward(d_wide, 2, Dual, Dual)]
fn wide(&x: &f64) -> f64 {
    x * x
}

struct S;
impl S {
    #[autodiff_reverse(d_method, Const, Active, Active, Active)]
    fn method(&self, _: f64, y: f64) -> f64 {
        y * y
    }
}

fn main() {
    let (mut dt, mut dx) = ((0.0, 0.0), 0.0);
    let _ = (wild(1.0, 2.0), d_wild(1.0, 2.0, 1.0));
    let _ = (tuple(&(1.0, 2.0)), d_tuple(&(1.0, 2.0), &mut dt, 1.0));
    let _ = (deref(&1.0), d_deref(&1.0, &mut dx, 1.0));
    #[cfg(check)]
    let _ = (binding((1.0, 2.0)), d_binding((1.0, 2.0), 1.0));
    let _ = (mut_ref(1.0, 2.0), d_mut_ref(1.0, 2.0, 1.0));
    #[cfg(check)]
    let _ = (dual(1.0, (1.0, 2.0)), d_dual(1.0, 1.0, (1.0, 2.0)));
    let _ = (dual_ref(1.0, &(1.0, 2.0)), d_dual_ref(1.0, 1.0, &(1.0, 2.0)));
    let _ = (wide(&1.0), d_wide(&1.0, &1.0, &1.0));
    let _ = (S.method(1.0, 2.0), S.d_method(1.0, 2.0, 1.0));
}
