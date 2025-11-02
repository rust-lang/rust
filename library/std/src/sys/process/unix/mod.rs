#[cfg_attr(any(target_os = "espidf", target_os = "horizon", target_os = "nuttx"), allow(unused))]
mod common;

cfg_select! {
    target_os = "fuchsia" => {
        mod fuchsia;
        use fuchsia as imp;
    }
    target_os = "vxworks" => {
        mod vxworks;
        use vxworks as imp;
    }
    any(target_os = "espidf", target_os = "horizon", target_os = "vita", target_os = "nuttx") => {
        mod unsupported;
        use unsupported as imp;
        pub use unsupported::output;
    }
    _ => {
        mod unix;
        use unix as imp;
    }
}

pub use imp::{ExitStatus, ExitStatusError, Process};

pub use self::common::{Command, CommandArgs, ExitCode, Stdio};
pub use crate::ffi::OsString as EnvKey;
