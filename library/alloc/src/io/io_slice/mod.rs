//! Target-dependant definition of `IoSlice` and `IoSliceMut`
//!
//! This is necessary to do it in `alloc` because other parts of the crate need
//! them even though they must have different layouts depending on the platform.
//!
//! However, we take great care to not leak platform-specific details and to not
//! link to any library here.

#![allow(unreachable_pub)]

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
