#![forbid(unsafe_op_in_unsafe_fn)]

mod error;

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
pub use error::{decode_error_kind, errno, error_string, is_interrupted};
pub use is_terminal::is_terminal;
pub use kernel_copy::{CopyState, kernel_copy};
