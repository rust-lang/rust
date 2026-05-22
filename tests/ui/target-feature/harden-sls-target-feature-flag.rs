//@ add-minicore
//@ revisions: by_flag by_feature
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ [by_flag]non-aux-compile-flags: -Zharden-sls=all
//@ [by_feature]non-aux-compile-flags: -Ctarget-feature=+harden-sls-ijmp,+harden-sls-ret
//@ [by_flag]build-pass
//@ [by_feature]check-fail
//@ ignore-backends: gcc
#![allow(non_camel_case_types)]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

//[by_feature]~? ERROR target feature `harden-sls-ijmp` cannot be enabled with `-Ctarget-feature`: use `harden-sls` compiler flag instead
//[by_feature]~? ERROR target feature `harden-sls-ret` cannot be enabled with `-Ctarget-feature`: use `harden-sls` compiler flag instead
