//@ revisions: one two three four
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//
//
//@ [one] check-fail
//@ [one] compile-flags: -C target-feature=+paca
//@ [two] check-fail
//@ [two] compile-flags: -C target-feature=-pacg,+pacg
//@ [three] check-fail
//@ [three] compile-flags: -C target-feature=+paca,+pacg,-paca
//@ [four] build-pass
//@ [four] compile-flags: -C target-feature=-paca,+pacg -C target-feature=+paca
//@ ignore-backends: gcc
//@ add-minicore
// FIXME(#147881): *disable* the features again for minicore as otherwise that will fail to build.
//@ minicore-compile-flags: -C target-feature=-pacg,-paca
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn main() {}

//[one]~? ERROR the target features paca, pacg must all be either enabled or disabled together
//[two]~? ERROR the target features paca, pacg must all be either enabled or disabled together
//[three]~? ERROR the target features paca, pacg must all be either enabled or disabled together
