//! Simple file-locking apis for each OS.
//!
//! This is not meant to be in the standard library, it does nothing with
//! green/native threading. This is just a bare-bones enough solution for
//! librustdoc, it is not production quality at all.

// cfg(bootstrap)
macro_rules! cfg_select_dispatch {
    ($($tokens:tt)*) => {
        #[cfg(bootstrap)]
        cfg_match! { $($tokens)* }

        #[cfg(not(bootstrap))]
        cfg_select! { $($tokens)* }
    };
}

cfg_select_dispatch! {
    target_os = "linux" => {
        mod linux;
        use linux as imp;
    }
    target_os = "redox" => {
        mod linux;
        use linux as imp;
    }
    unix => {
        mod unix;
        use unix as imp;
    }
    windows => {
        mod windows;
        use self::windows as imp;
    }
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

pub use imp::Lock;
