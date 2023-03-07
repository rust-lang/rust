#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![feature(link_cfg)]
#![feature(staged_api)]
#![feature(c_unwind)]
#![feature(cfg_target_abi)]
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

#[cfg(target_os = "android")]
cfg_if::cfg_if! {
    if #[cfg(feature = "llvm-libunwind")] {
        compile_error!("`llvm-libunwind` is not supported for Android targets");
    } else if #[cfg(feature = "system-llvm-libunwind")] {
        #[link(name = "unwind", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "unwind", cfg(not(target_feature = "crt-static")))]
        extern "C" {}
    } else {
        #[link(name = "gcc", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
        #[link(name = "gcc", cfg(not(target_feature = "crt-static")))]
        extern "C" {}
    }
}
// Android's unwinding library depends on dl_iterate_phdr in `libdl`.
#[cfg(target_os = "android")]
#[link(name = "dl", kind = "static", modifiers = "-bundle", cfg(target_feature = "crt-static"))]
#[link(name = "dl", cfg(not(target_feature = "crt-static")))]
extern "C" {}

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

#[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
#[link(name = "gcc_s")]
extern "C" {}

#[cfg(all(target_os = "openbsd", target_arch = "sparc64"))]
#[link(name = "gcc")]
extern "C" {}

#[cfg(all(target_os = "openbsd", not(target_arch = "sparc64")))]
#[link(name = "c++abi")]
extern "C" {}

#[cfg(any(target_os = "solaris", target_os = "illumos"))]
#[link(name = "gcc_s")]
extern "C" {}

#[cfg(target_os = "dragonfly")]
#[link(name = "gcc_pic")]
extern "C" {}

#[cfg(target_os = "haiku")]
#[link(name = "gcc_s")]
extern "C" {}

#[cfg(target_os = "nto")]
#[link(name = "gcc_s")]
extern "C" {}
