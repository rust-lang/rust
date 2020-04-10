#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![feature(link_cfg)]
#![feature(nll)]
#![feature(staged_api)]
#![feature(unwind_attributes)]
#![feature(static_nobundle)]
#![cfg_attr(not(target_env = "msvc"), feature(libc))]

cfg_if::cfg_if! {
    if #[cfg(target_env = "msvc")] {
        // no extra unwinder support needed
    } else if #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))] {
        // no unwinder on the system!
    } else {
        mod libunwind;
        pub use libunwind::*;
    }
}

#[cfg(target_env = "musl")]
#[link(name = "unwind", kind = "static", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(target_os = "redox")]
#[link(name = "gcc_eh", kind = "static-nobundle", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

// If any of our crates are dynamically linked then we need to use
// the shared libgcc_s-dw2-1.dll. This is required to support
// unwinding across DLL boundaries.
// If all of our crates are statically linked then we can get away
// with statically linking the libgcc unwinding code. This allows
// binaries to be redistributed without the libgcc_s-dw2-1.dll
// dependency, but unfortunately break unwinding across DLL
// boundaries when unwinding across FFI boundaries.
//
// Linking order matters a lot here!
#[cfg(all(not(bootstrap), target_os = "windows", target_env = "gnu", target_vendor = "pc"))]
#[link(name = "gcc_eh", kind = "static-nobundle")]
#[link(name = "pthread", kind = "static-nobundle")]
#[link(name = "gcc_s")]
extern "C" {}
