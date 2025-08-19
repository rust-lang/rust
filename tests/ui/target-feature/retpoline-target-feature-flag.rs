//@ add-core-stubs
//@ revisions: by_flag by_feature1 by_feature2 by_feature3
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ [by_flag]compile-flags: -Zretpoline

//@ [by_feature1]compile-flags: -Ctarget-feature=+retpoline-external-thunk
//@ [by_feature2]compile-flags: -Ctarget-feature=+retpoline-indirect-branches
//@ [by_feature3]compile-flags: -Ctarget-feature=+retpoline-indirect-calls
//@ [by_flag]build-pass
// For now this is just a warning.
//@ [by_feature1]build-pass
//@ [by_feature2]build-pass
//@ [by_feature3]build-pass
#![feature(no_core)]
#![no_core]
extern crate minicore;

//[by_feature1]~? WARN target feature `retpoline-external-thunk` cannot be enabled with `-Ctarget-feature`
//[by_feature2]~? WARN target feature `retpoline-indirect-branches` cannot be enabled with `-Ctarget-feature`
//[by_feature3]~? WARN target feature `retpoline-indirect-calls` cannot be enabled with `-Ctarget-feature`
