//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, nonstandard_style, missing_debug_implementations)]

// When documenting libstd we want to show unix/windows/linux/wasi modules as these are the "main
// modules" that are used across platforms, so all modules are enabled when `cfg(doc)` is set.
// This should help show platform-specific functionality in a hopefully cross-platform way in the
// documentation.
// Note that we deliberately avoid `cfg_if!` here to work around a rust-analyzer bug that would make
// `std::os` submodules unusable: https://github.com/rust-analyzer/rust-analyzer/issues/6038

#[cfg(doc)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::sys::unix_ext as unix;

#[cfg(doc)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::sys::windows_ext as windows;

#[cfg(doc)]
#[doc(cfg(target_os = "linux"))]
pub mod linux;

#[cfg(doc)]
pub use crate::sys::wasi_ext as wasi;

// If we're not documenting libstd then we just expose the main modules as we otherwise would.

#[cfg(not(doc))]
#[cfg(any(target_os = "redox", unix, target_os = "vxworks", target_os = "hermit"))]
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::sys::ext as unix;

#[cfg(not(doc))]
#[cfg(windows)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::sys::ext as windows;

#[cfg(not(doc))]
#[cfg(any(target_os = "linux", target_os = "l4re"))]
pub mod linux;

#[cfg(not(doc))]
#[cfg(target_os = "wasi")]
pub mod wasi;

#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "dragonfly")]
pub mod dragonfly;
#[cfg(target_os = "emscripten")]
pub mod emscripten;
#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
pub mod fortanix_sgx;
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

pub mod raw;
