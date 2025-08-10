#![forbid(unsafe_op_in_unsafe_fn)]

mod io_slice {
    cfg_select! {
        any(target_family = "unix", target_os = "hermit", target_os = "solid_asp3", target_os = "trusty") => {
            mod iovec;
            pub use iovec::*;
        }
        target_os = "windows" => {
            mod windows;
            pub use windows::*;
        }
        target_os = "wasi" => {
            mod wasi;
            pub use wasi::*;
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
        _ => {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

pub use io_slice::{IoSlice, IoSliceMut};
pub use is_terminal::is_terminal;

// Bare metal platforms usually have very small amounts of RAM
// (in the order of hundreds of KB)
pub const DEFAULT_BUF_SIZE: usize = if cfg!(target_os = "espidf") { 512 } else { 8 * 1024 };
