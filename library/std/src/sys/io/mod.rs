#![forbid(unsafe_op_in_unsafe_fn)]

mod error;

mod io_slice {
    cfg_select! {
        any(target_family = "unix", target_os = "hermit", target_os = "solid_asp3", target_os = "trusty", target_os = "wasi") => {
            mod iovec;
            pub use iovec::*;
        }
        target_os = "windows" => {
            mod windows;
            pub use windows::*;
        }
        target_os = "uefi" => {
            mod uefi;
            pub use uefi::*;
        }
        _ => {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

mod is_terminal {
    cfg_select! {
        any(target_family = "unix", target_os = "wasi") => {
            mod isatty;
            pub use isatty::*;
        }
        target_os = "windows" => {
            mod windows;
            pub use windows::*;
        }
        target_os = "hermit" => {
            mod hermit;
            pub use hermit::*;
        }
        target_os = "motor" => {
            mod motor;
            pub use motor::*;
        }
        _ => {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

mod kernel_copy;

#[cfg_attr(not(target_os = "linux"), allow(unused_imports))]
#[cfg(all(
    target_family = "unix",
    not(any(target_os = "dragonfly", target_os = "vxworks", target_os = "rtems"))
))]
pub use error::errno_location;
#[cfg_attr(not(target_os = "linux"), allow(unused_imports))]
#[cfg(any(
    all(target_family = "unix", not(any(target_os = "vxworks", target_os = "rtems"))),
    target_os = "wasi",
))]
pub use error::set_errno;
pub use error::{RawOsError, decode_error_kind, errno, error_string, is_interrupted};
pub use io_slice::{IoSlice, IoSliceMut};
pub use is_terminal::is_terminal;
pub use kernel_copy::{CopyState, kernel_copy};

// Bare metal platforms usually have very small amounts of RAM
// (in the order of hundreds of KB)
pub const DEFAULT_BUF_SIZE: usize = if cfg!(target_os = "espidf") { 512 } else { 8 * 1024 };
