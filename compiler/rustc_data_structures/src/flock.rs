//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

cfg_match! {
    cfg(target_os = "linux") => {
        mod linux;
        use linux as imp;
    }
    cfg(target_os = "redox") => {
        mod linux;
        use linux as imp;
    }
    cfg(unix) => {
        mod unix;
        use unix as imp;
    }
    cfg(windows) => {
        mod windows;
        use self::windows as imp;
    }
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

pub use imp::Lock;
