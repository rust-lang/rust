#![forbid(unsafe_op_in_unsafe_fn)]

mod io_slice {
    cfg_if::cfg_if! {
        if #[cfg(any(target_family = "unix", target_os = "hermit", target_os = "solid_asp3"))] {
            mod iovec;
            pub use iovec::*;
        } else if #[cfg(target_os = "windows")] {
            mod windows;
            pub use windows::*;
        } else if #[cfg(target_os = "wasi")] {
            mod wasi;
            pub use wasi::*;
        } else {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

mod is_terminal {
    cfg_if::cfg_if! {
        if #[cfg(any(target_family = "unix", target_os = "wasi"))] {
            mod isatty;
            pub use isatty::*;
        } else if #[cfg(target_os = "windows")] {
            mod windows;
            pub use windows::*;
        } else if #[cfg(target_os = "hermit")] {
            mod hermit;
            pub use hermit::*;
        } else {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

pub use io_slice::{IoSlice, IoSliceMut};
pub use is_terminal::is_terminal;
