#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    any(target_os = "windows", target_os = "uefi") => {
        mod wtf8;
        pub use wtf8::{Buf, Slice};
    }
    any(target_os = "motor") => {
        mod utf8;
        pub use utf8::{Buf, Slice};
    }
    _ => {
        mod bytes;
        pub use bytes::{Buf, Slice};
    }
}
