//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, nonstandard_style, missing_debug_implementations)]

pub mod raw;

// The code below could be written clearer using `cfg_if!`. However, the items below are
// publicly exported by `std` and external tools can have trouble analysing them because of the use
// of a macro that is not vendored by Rust and included in the toolchain.
// See https://github.com/rust-analyzer/rust-analyzer/issues/6038.

#[cfg(all(
    doc,
    not(any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    ))
))]
#[path = "."]
mod doc {
    // When documenting std we want to show the `unix`, `windows`, `linux` and `wasi`
    // modules as these are the "main modules" that are used across platforms,
    // so these modules are enabled when `cfg(doc)` is set.
    // This should help show platform-specific functionality in a hopefully cross-platform
    // way in the documentation.

    pub mod unix;

    pub mod linux;

    pub mod wasi;

    pub mod windows;
}
#[cfg(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
mod doc {
    // On certain platforms right now the "main modules" modules that are
    // documented don't compile (missing things in `libc` which is empty),
    // so just omit them with an empty module.

    #[unstable(issue = "none", feature = "std_internals")]
    pub mod unix {}

    #[unstable(issue = "none", feature = "std_internals")]
    pub mod linux {}

    #[unstable(issue = "none", feature = "std_internals")]
    pub mod wasi {}

    #[unstable(issue = "none", feature = "std_internals")]
    pub mod windows {}
}
#[cfg(doc)]
#[stable(feature = "os", since = "1.0.0")]
pub use doc::*;

#[cfg(not(doc))]
#[path = "."]
mod imp {
    // If we're not documenting std then we only expose modules appropriate for the
    // current platform.

    #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
    pub mod fortanix_sgx;

    #[cfg(target_os = "hermit")]
    #[path = "hermit/mod.rs"]
    pub mod unix;

    #[cfg(target_os = "android")]
    pub mod android;
    #[cfg(target_os = "dragonfly")]
    pub mod dragonfly;
    #[cfg(target_os = "emscripten")]
    pub mod emscripten;
    #[cfg(target_os = "espidf")]
    pub mod espidf;
    #[cfg(target_os = "freebsd")]
    pub mod freebsd;
    #[cfg(target_os = "fuchsia")]
    pub mod fuchsia;
    #[cfg(target_os = "haiku")]
    pub mod haiku;
    #[cfg(target_os = "illumos")]
    pub mod illumos;
    #[cfg(target_os = "ios")]
    pub mod ios;
    #[cfg(target_os = "l4re")]
    pub mod linux;
    #[cfg(target_os = "linux")]
    pub mod linux;
    #[cfg(target_os = "macos")]
    pub mod macos;
    #[cfg(target_os = "netbsd")]
    pub mod netbsd;
    #[cfg(target_os = "openbsd")]
    pub mod openbsd;
    #[cfg(target_os = "redox")]
    pub mod redox;
    #[cfg(target_os = "solaris")]
    pub mod solaris;
    #[cfg(unix)]
    pub mod unix;

    #[cfg(target_os = "vxworks")]
    pub mod vxworks;

    #[cfg(target_os = "wasi")]
    pub mod wasi;

    #[cfg(windows)]
    pub mod windows;
}
#[cfg(not(doc))]
#[stable(feature = "os", since = "1.0.0")]
pub use imp::*;
