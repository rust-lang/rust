//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, nonstandard_style, missing_debug_implementations)]
#![allow(unsafe_op_in_unsafe_fn)]

pub mod raw;

// # Important platforms

// We always want to show documentation for the most important platforms,
// so these are handled specially here.
//
// FIXME: On certain platforms compilation errors (due to empty `libc`),
// prevent this, so we substitute an unstable empty module.
#[cfg(doc)]
cfg_select! {
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    ) => {
        #[unstable(issue = "none", feature = "std_internals")]
        pub mod darwin {}

        #[unstable(issue = "none", feature = "std_internals")]
        pub mod unix {}

        #[unstable(issue = "none", feature = "std_internals")]
        pub mod linux {}

        #[unstable(issue = "none", feature = "std_internals")]
        pub mod wasi {}

        #[unstable(issue = "none", feature = "std_internals")]
        pub mod windows {}
    }
    _ => {
        // important platforms
        pub mod darwin;
        pub mod linux;
        pub mod unix;
        pub mod wasi;
        pub mod wasip2;
        pub mod windows;
    }
}
#[cfg(not(doc))] // to prevent double module declarations
cfg_select! {
    target_family = "unix" => {
        pub mod unix;
        #[cfg(target_vendor = "apple")]
        pub mod darwin;
        #[cfg(target_os = "linux")]
        pub mod linux;
    }
    target_family = "wasm" => {
        #[cfg(any(target_env = "p1", target_env = "p2"))]
        pub mod wasi;
        #[cfg(target_env = "p2")]
        pub mod wasip2;
    }
    target_family = "windows" => {
        pub mod windows;
    }
}

// # Special modules

#[cfg(any(
    unix,
    target_os = "hermit",
    target_os = "trusty",
    target_os = "wasi",
    target_os = "motor",
    doc
))]
pub mod fd;

#[cfg(any(target_os = "linux", target_os = "android", target_os = "cygwin", doc))]
mod net;

// # Ordinary platforms
// `cfg(doc)` not handled specially

#[cfg(target_os = "aix")]
pub mod aix;
#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "cygwin")]
pub mod cygwin;
#[cfg(target_os = "dragonfly")]
pub mod dragonfly;
#[cfg(target_os = "emscripten")]
pub mod emscripten;
#[cfg(target_os = "espidf")]
pub mod espidf;
#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub mod fortanix_sgx;
#[cfg(target_os = "freebsd")]
pub mod freebsd;
#[cfg(target_os = "fuchsia")]
pub mod fuchsia;
#[cfg(target_os = "haiku")]
pub mod haiku;
#[cfg(target_os = "hermit")]
pub mod hermit;
#[cfg(target_os = "horizon")]
pub mod horizon;
#[cfg(target_os = "hurd")]
pub mod hurd;
#[cfg(target_os = "illumos")]
pub mod illumos;
#[cfg(target_os = "ios")]
pub mod ios;
#[cfg(target_os = "l4re")]
pub mod l4re;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "motor")]
pub mod motor;
#[cfg(target_os = "netbsd")]
pub mod netbsd;
#[cfg(target_os = "nto")]
pub mod nto;
#[cfg(target_os = "nuttx")]
pub mod nuttx;
#[cfg(target_os = "openbsd")]
pub mod openbsd;
#[cfg(target_os = "redox")]
pub mod redox;
#[cfg(target_os = "rtems")]
pub mod rtems;
#[cfg(target_os = "solaris")]
pub mod solaris;
#[cfg(target_os = "solid_asp3")]
pub mod solid;
#[cfg(target_os = "trusty")]
pub mod trusty;
#[cfg(target_os = "uefi")]
pub mod uefi;
#[cfg(target_os = "vita")]
pub mod vita;
#[cfg(target_os = "vxworks")]
pub mod vxworks;
#[cfg(target_os = "xous")]
pub mod xous;
