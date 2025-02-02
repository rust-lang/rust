//@ add-core-stubs
//@ revisions: by_flag by_feature
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ [by_flag]compile-flags: -Zharden-sls=all
//@ [by_feature]compile-flags: -Ctarget-feature=+harden-sls-ijmp,+harden-sls-ret
//@ [by_flag]build-pass
// For now this is just a warning.
//@ [by_feature]build-pass
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

//[by_feature]~? WARN target feature `harden-sls-ijmp` cannot be enabled with `-Ctarget-feature`: use `harden-sls` compiler flag instead
//[by_feature]~? WARN target feature `harden-sls-ret` cannot be enabled with `-Ctarget-feature`: use `harden-sls` compiler flag instead
