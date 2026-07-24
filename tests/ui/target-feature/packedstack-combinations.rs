//@ add-minicore
//@ revisions: wrong_arch only_packedstack backchain_attr backchain_cli need_softfloat with_softfloat
//@ compile-flags: -Zpacked-stack --crate-type=rlib
//@ ignore-backends: gcc

//@ [wrong_arch] compile-flags: --target=x86_64-unknown-linux-gnu
//@ [wrong_arch] should-fail
//@ [wrong_arch] needs-llvm-components: x86

//@ [only_packedstack] compile-flags: --target=s390x-unknown-linux-gnu
//@ [only_packedstack] build-pass
//@ [only_packedstack] needs-llvm-components: systemz

//@ [backchain_attr] compile-flags: --target=s390x-unknown-linux-gnu
//@ [backchain_attr] build-fail
//@ [backchain_attr] needs-llvm-components: systemz

//@ [backchain_cli] compile-flags: -Ctarget-feature=+backchain --target=s390x-unknown-linux-gnu
//@ [backchain_cli] should-fail
//@ [backchain_cli] needs-llvm-components: systemz

//@ [need_softfloat] compile-flags: -Ctarget-feature=+backchain
//@ [need_softfloat] compile-flags: --target=s390x-unknown-linux-gnu
//@ [need_softfloat] should-fail
//@ [need_softfloat] needs-llvm-components: systemz

//@ [with_softfloat] compile-flags: -Ctarget-feature=+backchain
//@ [with_softfloat] compile-flags: --target=s390x-unknown-none-softfloat
//@ [with_softfloat] build-pass
//@ [with_softfloat] needs-llvm-components: systemz

#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
#[cfg_attr(backchain_attr, target_feature(enable = "backchain"))]
pub fn test() {}

//[wrong_arch]~? ERROR `-Zpacked-stack` is only supported on s390x
//[backchain_cli]~? ERROR `-Zpacked-stack` is incompatible with `backchain` target feature
//[backchain_attr]~? ERROR `-Zpacked-stack` is incompatible with `backchain` target feature
//[need_softfloat]~? ERROR `-Zpacked-stack` is incompatible with `backchain` target feature
