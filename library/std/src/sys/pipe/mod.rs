//! Platform-dependent pipe abstraction.

#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    unix => {
        mod unix;
        pub use self::unix::*;
    }
    windows => {
        mod windows;
        pub use self::windows::*;
    }
    _ => {
        mod unsupported;
        pub use self::unsupported::*;
    }
}
