//! Support for "weak linkage" to symbols on Unix
//!
//! Some I/O operations we do in std require newer versions of OSes but we need
//! to maintain binary compatibility with older releases for now. In order to
//! use the new functionality when available we use this module for detection.
//!
//! One option to use here is weak linkage, but that is unfortunately only
//! really workable with ELF. Otherwise, use dlsym to get the symbol value at
//! runtime. This is also done for compatibility with older versions of glibc,
//! and to avoid creating dependencies on GLIBC_PRIVATE symbols. It assumes that
//! we've been dynamically linked to the library the symbol comes from, but that
//! is currently always the case for things like libpthread/libc.
//!
//! A long time ago this used weak linkage for the __pthread_get_minstack
//! symbol, but that caused Debian to detect an unnecessarily strict versioned
//! dependency on libc6 (#23628) because it is GLIBC_PRIVATE. We now use `dlsym`
//! for a runtime lookup of that symbol to avoid the ELF versioned dependency.

#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    // On non-ELF targets, use the dlsym approximation of weak linkage.
    target_vendor = "apple" => {
        mod dlsym;
        pub(crate) use dlsym::weak;
    }

    // Some targets don't need and support weak linkage at all...
    target_os = "espidf" => {}

    // ... but ELF targets support true weak linkage.
    _ => {
        // There are a variety of `#[cfg]`s controlling which targets are involved in
        // each instance of `weak!`. Rather than trying to unify all of
        // that, we'll just allow that some unix targets don't use this macro at all.
        #[cfg_attr(not(target_os = "linux"), allow(unused_macros, dead_code))]
        mod weak_linkage;
        #[cfg_attr(not(target_os = "linux"), allow(unused_imports))]
        pub(crate) use weak_linkage::weak;
    }
}

// GNU/Linux needs the `dlsym` variant to avoid linking to private glibc symbols.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod dlsym;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) use dlsym::weak as dlsym;

#[cfg(any(target_os = "android", target_os = "linux"))]
mod syscall;
#[cfg(any(target_os = "android", target_os = "linux"))]
pub(crate) use syscall::syscall;
