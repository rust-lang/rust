#![feature(compiler_builtins_lib)]
#![feature(lang_items)]
#![allow(bad_style)]
#![allow(unused_imports)]
#![no_std]

extern crate compiler_builtins as builtins;

#[cfg(all(
    target_arch = "arm",
    not(any(target_env = "gnu", target_env = "musl")),
    target_os = "linux",
    test
))]
extern crate utest_cortex_m_qemu;

#[cfg(all(
    target_arch = "arm",
    not(any(target_env = "gnu", target_env = "musl")),
    target_os = "linux",
    test
))]
#[macro_use]
extern crate utest_macros;

#[cfg(all(
    target_arch = "arm",
    not(any(target_env = "gnu", target_env = "musl")),
    target_os = "linux",
    test
))]
macro_rules! panic { // overrides `panic!`
    ($($tt:tt)*) => {
        upanic!($($tt)*);
    };
}

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
