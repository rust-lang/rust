//@ assembly-output: ptx-linker
//@ compile-flags: --target=nvptx64-nvidia-cuda --crate-type cdylib -C target-cpu=sm_80
//@ needs-llvm-components: nvptx
//@ build-pass
//@ ignore-backends: gcc
#![feature(no_core, rustc_attrs)]
#![no_core]
#![allow(dead_code)]

#[rustc_builtin_macro]
#[macro_export]
macro_rules! compile_error {
    ($msg:expr $(,)?) => {{ /* compiler built-in */ }};
}

// -Ctarget-cpu=sm_80 directly enables sm_80 and ptx70
#[cfg(not(all(target_feature = "sm_80", target_feature = "ptx70")))]
compile_error!("direct target features not enabled");

// -Ctarget-cpu=sm_80 implies all earlier sm_* and ptx* features.
#[cfg(not(all(
    target_feature = "sm_60",
    target_feature = "sm_70",
    target_feature = "ptx50",
    target_feature = "ptx60",
)))]
compile_error!("implied target features not enabled");

// -Ctarget-cpu=sm_80 implies all earlier sm_* and ptx* features.
#[cfg(target_feature = "ptx71")]
compile_error!("sm_80 requires only ptx70, but ptx71 enabled");
