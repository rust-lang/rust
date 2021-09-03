//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, nonstandard_style, missing_debug_implementations)]

pub mod raw;

// The code below could be written clearer using `cfg_if!`. However, the items below are
// publicly exported by `std` and external tools can have trouble analysing them because of the use
// of a macro that is not vendored by Rust and included in the toolchain.
// See https://github.com/rust-analyzer/rust-analyzer/issues/6038.

// On certain platforms right now the "main modules" modules that are
// documented don't compile (missing things in `libc` which is empty),
// so just omit them with an empty module and add the "unstable" attribute.

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub mod fortanix_sgx;

// Unix, linux, wasi and windows are handled a bit differently.
#[cfg(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
#[unstable(issue = "none", feature = "std_internals")]
pub mod unix {}
#[cfg(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
#[unstable(issue = "none", feature = "std_internals")]
pub mod linux {}
#[cfg(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
#[unstable(issue = "none", feature = "std_internals")]
pub mod wasi {}
#[cfg(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
#[unstable(issue = "none", feature = "std_internals")]
pub mod windows {}

// unix
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
#[cfg(all(not(doc), target_os = "hermit"))]
#[path = "hermit/mod.rs"]
pub mod unix;
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
#[cfg(any(unix, doc))]
pub mod unix;

// linux
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
#[cfg(any(target_os = "linux", target_os = "l4re", doc))]
pub mod linux;

// wasi
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
#[cfg(any(target_os = "wasi", doc))]
pub mod wasi;

// windows
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
#[cfg(any(windows, doc))]
pub mod windows;

// Others.
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

#[cfg(any(unix, target_os = "wasi", doc))]
mod fd;
