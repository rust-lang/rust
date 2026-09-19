//@ aux-build:pauth.rs
//@ revisions: ok ok_reverse_order ok_last_one_wins error_generated
//@ compile-flags: --target=aarch64-unknown-linux-pauthtest
//@ [ok] compile-flags: -Zpointer-authentication=+calls,+init-fini
//@ [ok] check-pass
//@ [ok_last_one_wins] compile-flags: -Zpointer-authentication=-calls,+calls,-init-fini,+init-fini
//@ [ok_last_one_wins] check-pass
//@ [ok_reverse_order] compile-flags: -Zpointer-authentication=+init-fini,+calls
//@ [ok_reverse_order] check-pass
//@ [error_generated] compile-flags: -Zpointer-authentication=+calls,-init-fini
//@ needs-llvm-components: aarch64
//@ only-pauthtest

#![feature(no_core)]
//[error_generated]~^ ERROR mixing `-Zpointer-authentication` will cause an ABI mismatch in crate
//`incompatible_pauth`
#![crate_type = "rlib"]
#![no_core]

extern crate pauth;
