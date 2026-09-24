//! Ensure `backchain` cannot be enabled via `-Ctarget-feature` on s390x.
//@ compile-flags: --crate-type=lib --target=s390x-unknown-linux-gnu
//@ compile-flags: -Ctarget-feature=+backchain
//@ needs-llvm-components: systemz
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]

//~? ERROR target feature `backchain` cannot be enabled with `-Ctarget-feature`: use -Cforce-frame-pointers instead
