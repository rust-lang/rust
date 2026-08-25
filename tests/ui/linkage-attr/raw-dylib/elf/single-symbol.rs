//@ only-elf
//@ needs-dynamic-linking
// FIXME(raw_dylib_elf): Debug the failures on other targets.
//@ only-gnu
//@ only-x86_64
//@ revisions: with without

//@ [without] build-fail
//@ [without] regex-error-pattern:error: linking with `.*` failed
//@ [without] dont-check-compiler-stderr

//@ [with] build-pass

//! Ensures that linking fails when there's an undefined symbol,
//! and that it does succeed with raw-dylib.

#![feature(raw_dylib_elf)]
#![allow(incomplete_features)]

#[cfg_attr(with, link(name = "rawdylibbutforcats", kind = "raw-dylib"))]
#[cfg_attr(without, link(name = "rawdylibbutforcats"))]
unsafe extern "C" {
  safe fn meooooooooooooooow();
}

fn main() {
  meooooooooooooooow();
}

//[without]~? ERROR linking with `
