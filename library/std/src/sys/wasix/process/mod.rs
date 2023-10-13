pub use self::process_common::{Command, CommandArgs, ExitCode, Stdio, StdioPipes};
pub use self::process_inner::{ExitStatus, ExitStatusError, Process};
pub use crate::ffi::OsString as EnvKey;
pub use crate::sys_common::process::CommandEnvs;

#[path = "../../unix/process/process_common.rs"]
#[cfg_attr(any(target_os = "espidf", target_os = "horizon"), allow(unused))]
mod process_common;

#[path = "process_wasix.rs"]
mod process_inner;
