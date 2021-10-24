#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![feature(link_cfg)]
#![feature(native_link_modifiers)]
#![feature(native_link_modifiers_bundle)]
#![feature(nll)]
#![feature(staged_api)]
#![feature(c_unwind)]
#![cfg_attr(not(target_env = "msvc"), feature(libc))]

cfg_if::cfg_if! {
    if #[cfg(target_env = "msvc")] {
        // Windows MSVC no extra unwinder support needed
    } else if #[cfg(any(
        target_os = "l4re",
        target_os = "none",
        target_os = "espidf",
    ))] {
        // These "unix" family members do not have unwinder.
        // Note this also matches x86_64-unknown-none-linuxkernel.
    } else if #[cfg(any(
        unix,
        windows,
        target_os = "psp",
        target_os = "solid_asp3",
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
cfg_if::cfg_if! {
    if #[cfg(all(feature = "llvm-libunwind", feature = "system-llvm-libunwind"))] {
        compile_error!("`llvm-libunwind` and `system-llvm-libunwind` cannot be enabled at the same time");
    } else if #[cfg(feature = "llvm-libunwind")] {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle")]
        extern "C" {}
    } else if #[cfg(feature = "system-llvm-libunwind")] {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "unwind", cfg(not(target_feature = "crt-static")))]
        extern "C" {}
    } else {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
        extern "C" {}
    }
}

// When building with crt-static, we get `gcc_eh` from the `libc` crate, since
// glibc needs it, and needs it listed later on the linker command line. We
// don't want to duplicate it here.
#[cfg(all(
    target_os = "linux",
    any(target_env = "gnu", target_env = "uclibc"),
    not(feature = "llvm-libunwind"),
    not(feature = "system-llvm-libunwind")
))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(all(
    target_os = "linux",
    any(target_env = "gnu", target_env = "uclibc"),
    not(feature = "llvm-libunwind"),
    feature = "system-llvm-libunwind"
))]
#[link(name = "unwind", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(target_os = "redox")]
#[link(name = "gcc_eh", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
#[link(name = "unwind", kind = "static", modifiers = "-bundle")]
extern "C" {}
