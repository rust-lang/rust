#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    any(target_os = "windows", target_os = "uefi") => {
        mod wtf8;
        pub(crate) use wtf8::{Buf, BytesFlavour, Slice};
    }
    any(target_os = "motor") => {
        mod utf8;
        pub(crate) use utf8::{Buf, BytesFlavour, Slice};
    }
    _ => {
        mod bytes;
        pub(crate) use bytes::{Buf, BytesFlavour, Slice};
    }
}
