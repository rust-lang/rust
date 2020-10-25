pub use self::process_common::{Command, CommandArgs, ExitCode, Stdio, StdioPipes};
pub use self::process_inner::{ExitStatus, Process};
pub use crate::ffi::OsString as EnvKey;
pub use crate::sys_common::process::CommandEnvs;

#[path = "../../unix/process/process_common.rs"]
mod process_common;
#[path = "process_vxworks.rs"]
mod process_inner;
