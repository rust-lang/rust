//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, nonstandard_style, missing_debug_implementations)]

pub mod raw;

cfg_if::cfg_if! {
    if #[cfg(all(doc, not(any(target_os = "hermit",
                        all(target_arch = "wasm32", not(target_os = "wasi")),
                        all(target_vendor = "fortanix", target_env = "sgx")))))]{
        // When documenting std we want to show the `unix`, `windows`, `linux` and `wasi`
        // modules as these are the "main modules" that are used across platforms,
        // so these modules are enabled when `cfg(doc)` is set.
        // This should help show platform-specific functionality in a hopefully cross-platform
        // way in the documentation.

        #[stable(feature = "rust1", since = "1.0.0")]
        pub use crate::sys::unix_ext as unix;

        pub mod linux;

        #[stable(feature = "wasi_ext_doc", since = "1.35.0")]
        pub use crate::sys::wasi_ext as wasi;

        pub mod windows;
    } else if #[cfg(doc)] {
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
    } else {
        // If we're not documenting std then we only expose modules appropriate for the
        // current platform.

        #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
        pub mod fortanix_sgx;

        #[cfg(target_os = "hermit")]
        mod hermit;
        #[cfg(target_os = "hermit")]
        pub use hermit as unix;

        #[cfg(unix)]
        #[stable(feature = "rust1", since = "1.0.0")]
        pub use crate::sys::ext as unix;
        #[cfg(target_os = "android")]
        pub mod android;
        #[cfg(target_os = "dragonfly")]
        pub mod dragonfly;
        #[cfg(target_os = "emscripten")]
        pub mod emscripten;
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

        #[cfg(target_os = "vxworks")]
        pub mod vxworks;

        #[cfg(target_os = "wasi")]
        pub mod wasi;

        #[cfg(windows)]
        pub mod windows;
    }
}
