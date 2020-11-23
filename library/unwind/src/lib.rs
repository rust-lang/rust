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
        // Windows MSVC no extra unwinder support needed
    } else if #[cfg(any(
        target_os = "l4re",
        target_os = "none",
    ))] {
        // These "unix" family members do not have unwinder.
        // Note this also matches x86_64-linux-kernel.
    } else if #[cfg(any(
        unix,
        windows,
        target_os = "psp",
        all(target_vendor = "fortanix", target_env = "sgx"),
    ))] {
        mod libunwind;
        pub use libunwind::*;
    } else {
        // no unwinder on the system!
        // - wasm32 (not emscripten, which is "unix" family)
        // - os=none ("bare metal" targets)
        // - os=hermit
        // - os=uefi
        // - os=cuda
        // - nvptx64-nvidia-cuda
        // - Any new targets not listed above.
    }
}

#[cfg(target_env = "musl")]
#[link(name = "unwind", kind = "static", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

// When building with crt-static, we get `gcc_eh` from the `libc` crate, since
// glibc needs it, and needs it listed later on the linker command line. We
// don't want to duplicate it here.
#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    not(feature = "llvm-libunwind"),
    not(feature = "system-llvm-libunwind")
))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    not(feature = "llvm-libunwind"),
    feature = "system-llvm-libunwind"
))]
#[link(name = "unwind", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(target_os = "redox")]
#[link(name = "gcc_eh", kind = "static-nobundle", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
#[link(name = "unwind", kind = "static-nobundle")]
extern "C" {}
