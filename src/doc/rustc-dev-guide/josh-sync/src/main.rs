use clap::Parser;
use crate::sync::GitSync;

mod sync;

#[derive(clap::Parser)]
enum Args {
    /// Pull changes from the main `rustc` repository.
    /// This creates new commits that should be then merged into `rustc-dev-guide`.
    RustcPull,
    /// Push changes from `rustc-dev-guide` to the given `branch` of a `rustc` fork under the given
    /// GitHub `username`.
    /// The pushed branch should then be merged into the `rustc` repository.
    RustcPush {
        branch: String,
        github_username: String
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let sync = GitSync::from_current_dir()?;
    match args {
        Args::RustcPull => {
            sync.rustc_pull(None)?;
        }
        Args::RustcPush { github_username, branch } => {
            sync.rustc_push(github_username, branch)?;
        }
    }
    Ok(())
}
