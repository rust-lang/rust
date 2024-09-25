use crate::command::Command;
use crate::env_var;

/// Returns a command that can be used to invoke cargo. The cargo is provided by compiletest
/// through the `CARGO` env var.
pub fn cargo() -> Command {
    Command::new(env_var("CARGO"))
}
