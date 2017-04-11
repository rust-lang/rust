#![feature(compiler_builtins_lib)]
#![feature(i128_type)]
#![cfg_attr(all(target_arch = "arm",
                not(any(target_env = "gnu", target_env = "musl")),
                target_os = "linux",
                test), no_std)]

// FIXME(#137)
// FIXME(#158)
#[cfg(not(any(target_arch = "mips", windows)))]
include!(concat!(env!("OUT_DIR"), "/modti3.rs"));
