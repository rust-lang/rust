use crate::utils::cmd::{cmd, TrackedCommand};
use std::path::Path;

/// Command that checks whether git works at the given `directory`.
pub fn cmd_works(directory: &Path) -> TrackedCommand {
    cmd_git(Some(directory)).arg("rev-parse")
}

/// Command that finds the currently checked out SHA at the given `directory`.
pub fn cmd_get_current_sha(directory: &Path) -> TrackedCommand {
    cmd_git(Some(directory)).arg("rev-parse").arg("HEAD")
}

pub fn cmd_git(directory: Option<&Path>) -> TrackedCommand {
    let mut cmd = cmd("git");
    if let Some(directory) = directory {
        cmd = cmd.current_dir(directory);
    }
    cmd
}
