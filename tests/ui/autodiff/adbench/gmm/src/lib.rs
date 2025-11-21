//@ revisions: LooseTypes default
//@[LooseTypes] compile-flags: -Zautodiff=Enable,LooseTypes -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat --crate-type=staticlib
//@[default] compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat --crate-type=staticlib
//@[LooseTypes] build-pass
//@[default] build-fail
//@ dont-check-compiler-stderr
//@ dont-check-compiler-stdout
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]
pub mod safe;
pub mod r#unsafe;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Wishart {
    pub gamma: f64,
    pub m: i32,
}
