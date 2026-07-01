//@ compile-flags: --target=nvptx64-nvidia-cuda --crate-type cdylib -C target-cpu=sm_90
//@ needs-llvm-components: nvptx
//@ check-pass
//@ ignore-backends: gcc
#![feature(no_core, rustc_attrs)]
#![no_core]
#![allow(dead_code)]

#[rustc_builtin_macro]
#[macro_export]
macro_rules! compile_error {
    ($msg:expr $(,)?) => {{ /* compiler built-in */ }};
}

// -Ctarget-cpu=sm_90 directly enables sm_90 and ptx78
#[cfg(not(all(target_feature = "sm_90", target_feature = "ptx78")))]
compile_error!("direct target features not enabled");

// -Ctarget-cpu=sm_90 implies all earlier sm_* and ptx* features.
#[cfg(not(all(
    target_feature = "sm_70",
    target_feature = "sm_80",
    target_feature = "ptx71",
    target_feature = "ptx74",
)))]
compile_error!("implied target features not enabled");

// -Ctarget-cpu=sm_90 implies all earlier sm_* and ptx* features.
#[cfg(target_feature = "ptx80")]
compile_error!("sm_90 requires only ptx78, but ptx80 enabled");
