#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    unix => {
        mod unix;
        pub use unix::{AnonPipe, pipe};
    }
    windows => {
        mod windows;
        pub use windows::{AnonPipe, pipe};
    }
    _ => {
        mod unsupported;
        pub use unsupported::{AnonPipe, pipe};
    }
}
