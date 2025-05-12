use crate::command::Command;
use crate::env_var;
use crate::util::set_host_compiler_dylib_path;

/// Returns a command that can be used to invoke cargo. The cargo is provided by compiletest
/// through the `CARGO` env var.
pub fn cargo() -> Command {
    let mut cmd = Command::new(env_var("CARGO"));
    set_host_compiler_dylib_path(&mut cmd);
    cmd
}
