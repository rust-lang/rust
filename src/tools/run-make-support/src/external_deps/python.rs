use crate::command::Command;
use crate::env::env_var;

/// Obtain path of python as provided by the `PYTHON` environment variable. It is up to the caller
/// to document and check if the python version is compatible with its intended usage.
#[track_caller]
#[must_use]
pub fn python_command() -> Command {
    let python_path = env_var("PYTHON");
    Command::new(python_path)
}
