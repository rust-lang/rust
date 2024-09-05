use crate::command::Command;
use crate::env_var;

/// Returns a command that can be used to invoke Cargo.
pub fn cargo() -> Command {
    Command::new(env_var("BOOTSTRAP_CARGO"))
}
