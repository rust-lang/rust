#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    any(target_os = "windows", target_os = "uefi") => {
        mod wtf8;
        pub use wtf8::{Buf, Slice};
    }
    _ => {
        mod bytes;
        pub use bytes::{Buf, Slice};
    }
}
