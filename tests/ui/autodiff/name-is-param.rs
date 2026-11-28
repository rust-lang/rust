//! Regression test for <https://github.com/rust-lang/rust/issues/150330>.
//! The generated function calls the autodiff intrinsic with a path to itself and to the original
//! function. A parameter with the same name used to hide that path: codegen then ICEd, or type
//! checking rejected the cast of the original function.

//@ revisions: check enzyme
//@[check] check-pass
//@[enzyme] needs-enzyme
//@[enzyme] compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
//@[enzyme] no-prefer-dynamic
//@[enzyme] build-pass
//@ proc-macro: add_autodiff.rs

#![feature(autodiff)]

extern crate add_autodiff;

use std::autodiff::{autodiff_forward, autodiff_reverse};

use add_autodiff::add_autodiff;

// The code from the issue.
macro_rules! demo {
    () => {
        #[std::autodiff::autodiff_reverse(x, Active, Active)]
        fn f(x: f64) -> f64 {
            x * x
        }
    };
}
demo!();

// The same without a macro.
#[autodiff_reverse(y, Active, Active)]
fn g(y: f64) -> f64 {
    y * y
}

// The name comes from a `macro_rules!` caller.
macro_rules! named {
    ($name:ident) => {
        #[autodiff_reverse($name, Active, Active)]
        fn h(z: f64) -> f64 {
            z * z
        }
    };
}
named!(z);

// A proc macro adds the attribute, so the name and the parameter have different contexts.
#[add_autodiff(w)]
fn k(w: f64) -> f64 {
    w * w
}

// A function inside the body of another function.
fn m() -> f64 {
    #[autodiff_reverse(v, Active, Active)]
    fn local(v: f64) -> f64 {
        v * v
    }
    local(1.0) + v(1.0, 1.0).0
}

// A generic function: the call passes the generic arguments on.
#[autodiff_reverse(u, Active, Active)]
fn generic<T>(u: f64) -> f64 {
    u * u
}

// A parameter with the name of the original function.
#[autodiff_reverse(dn, Active, Active)]
fn n(n: f64) -> f64 {
    n * n
}

// The names of generated parameters.
#[autodiff_reverse(dx_0, Duplicated, Active)]
fn p(x: &f64) -> f64 {
    *x * *x
}

#[autodiff_reverse(dret, Active, Active)]
fn q(x: f64) -> f64 {
    x * x
}

#[autodiff_forward(bx_1, 2, Dual, Dual)]
fn r(x: f64) -> f64 {
    x * x
}

// In a method the call is `Self::name`, which no parameter can hide.
struct S;
impl S {
    #[autodiff_reverse(a, Const, Active, Active)]
    fn s(&self, a: f64) -> f64 {
        a * a
    }
}

fn main() {
    let mut dx = 0.0;
    let _ = (f(1.0), x(1.0, 1.0));
    let _ = (g(1.0), y(1.0, 1.0));
    let _ = (h(1.0), z(1.0, 1.0));
    let _ = (k(1.0), w(1.0, 1.0));
    let _ = m();
    let _ = (generic::<u8>(1.0), u::<u8>(1.0, 1.0));
    let _ = (n(1.0), dn(1.0, 1.0));
    let _ = (p(&1.0), dx_0(&1.0, &mut dx, 1.0));
    let _ = (q(1.0), dret(1.0, 1.0));
    let _ = (r(1.0), bx_1(1.0, 1.0, 1.0));
    let _ = (S.s(1.0), S.a(1.0, 1.0));
}
