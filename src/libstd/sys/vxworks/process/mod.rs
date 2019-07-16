pub use self::process_common::{Command, ExitStatus, ExitCode, Stdio, StdioPipes};
pub use self::process_inner::Process;

mod process_common;
#[path = "process_vxworks.rs"]
mod process_inner;
mod rtp;
