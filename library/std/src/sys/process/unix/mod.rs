#[cfg_attr(any(target_os = "espidf", target_os = "horizon", target_os = "nuttx"), allow(unused))]
mod common;

cfg_if::cfg_if! {
    if #[cfg(target_os = "fuchsia")] {
        mod fuchsia;
        use fuchsia as imp;
    } else if #[cfg(target_os = "vxworks")] {
        mod vxworks;
        use vxworks as imp;
    } else if #[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita", target_os = "nuttx"))] {
        mod unsupported;
        use unsupported as imp;
        pub use unsupported::output;
    } else {
        mod unix;
        use unix as imp;
    }
}

pub use imp::{ExitStatus, ExitStatusError, Process};

pub use self::common::{Command, CommandArgs, ExitCode, Stdio, StdioPipes};
pub use crate::ffi::OsString as EnvKey;
