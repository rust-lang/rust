pub use self::process_common::{Command, ExitStatus, ExitCode, Stdio, StdioPipes};
pub use self::process_inner::Process;
pub use crate::ffi::OsString as EnvKey;

mod process_common;
#[path = "process_vxworks.rs"]
mod process_inner;
mod rtp;
