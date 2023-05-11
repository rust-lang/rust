//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

cfg_if! {
    if #[cfg(target_os = "linux")] {
        mod linux;
        use linux as imp;
    } else if #[cfg(unix)] {
        mod unix;
        use unix as imp;
    } else if #[cfg(windows)] {
        mod windows;
        use self::windows as imp;
    } else {
        mod unsupported;
        use unsupported as imp;
    }
}

pub use imp::Lock;
