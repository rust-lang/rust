//@ add-minicore
//@ revisions: by_flag by_feature1 by_feature2 by_feature3
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ [by_flag]compile-flags: -Zretpoline

//@ [by_feature1]non-aux-compile-flags: -Ctarget-feature=+retpoline-external-thunk
//@ [by_feature2]non-aux-compile-flags: -Ctarget-feature=+retpoline-indirect-branches
//@ [by_feature3]non-aux-compile-flags: -Ctarget-feature=+retpoline-indirect-calls
//@ [by_flag]build-pass
//@ [by_feature1]check-fail
//@ [by_feature2]check-fail
//@ [by_feature3]check-fail
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]
extern crate minicore;

//[by_feature1]~? ERROR target feature `retpoline-external-thunk` cannot be enabled with `-Ctarget-feature`
//[by_feature2]~? ERROR target feature `retpoline-indirect-branches` cannot be enabled with `-Ctarget-feature`
//[by_feature3]~? ERROR target feature `retpoline-indirect-calls` cannot be enabled with `-Ctarget-feature`
