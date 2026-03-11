//@ add-minicore
//@ compile-flags: -Z instrument-function=fentry -Copt-level=0
//
//@ revisions: x86_64-linux
//@[x86_64-linux] compile-flags: --target=x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86
//
//@ revisions: x86-linux
//@[x86-linux] compile-flags: --target=i686-unknown-linux-gnu
//@[x86-linux] needs-llvm-components: x86
//
//@ revisions: s390x-linux
//@[s390x-linux] compile-flags: --target=s390x-unknown-linux-gnu
//@[s390x-linux] needs-llvm-components: systemz

#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK: attributes #{{.*}} "fentry-call"="true"
pub fn foo() {}
