//@ aux-build:generic-const-exprs-dep.rs
//@ compile-flags: --crate-type=lib
//@ failure-status: 101
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0

extern crate generic_const_exprs_dep;
use generic_const_exprs_dep::{Tr, Foo};

pub fn build(bar: ()) {
    Foo::foo(bar)
}

//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
//~? ERROR Encountered anon const with inference variable args but no error reported
