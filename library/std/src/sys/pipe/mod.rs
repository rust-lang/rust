#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    unix => {
        mod unix;
        pub use unix::{Pipe, pipe};
    }
    windows => {
        mod windows;
        pub use windows::{Pipe, pipe};
    }
    target_os = "motor" => {
        mod motor;
        pub use motor::{Pipe, pipe};
    }
    _ => {
        mod unsupported;
        pub use unsupported::{Pipe, pipe};
    }
}
