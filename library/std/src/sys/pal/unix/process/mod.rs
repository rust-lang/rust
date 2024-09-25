pub use self::process_common::{Command, CommandArgs, ExitCode, Stdio, StdioPipes};
pub use self::process_inner::{ExitStatus, ExitStatusError, Process};
pub use crate::ffi::OsString as EnvKey;

#[cfg_attr(any(target_os = "espidf", target_os = "horizon", target_os = "nuttx"), allow(unused))]
mod process_common;

#[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita", target_os = "nuttx"))]
mod process_unsupported;

cfg_if::cfg_if! {
    if #[cfg(target_os = "fuchsia")] {
        #[path = "process_fuchsia.rs"]
        mod process_inner;
        mod zircon;
    } else if #[cfg(target_os = "vxworks")] {
        #[path = "process_vxworks.rs"]
        mod process_inner;
    } else if #[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita", target_os = "nuttx"))] {
        mod process_inner {
            pub use super::process_unsupported::*;
        }
    } else {
        #[path = "process_unix.rs"]
        mod process_inner;
    }
}
