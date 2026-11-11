//@ compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//@ build-pass

// Check that differentiating functions with ZST args does not break

#![feature(autodiff)]

#[core::autodiff::autodiff_forward(fd_inner, Const, Dual)]
fn f(_zst: (), _x: &mut f64) {}

fn fd(x: &mut f64, xd: &mut f64) {
    fd_inner((), x, xd);
}

fn main() {}
