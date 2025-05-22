use clap::Parser;

use crate::sync::{GitSync, RustcPullError};

mod sync;

#[derive(clap::Parser)]
enum Args {
    /// Pull changes from the main `rustc` repository.
    /// This creates new commits that should be then merged into `rustc-dev-guide`.
    RustcPull,
    /// Push changes from `rustc-dev-guide` to the given `branch` of a `rustc` fork under the given
    /// GitHub `username`.
    /// The pushed branch should then be merged into the `rustc` repository.
    RustcPush { branch: String, github_username: String },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let sync = GitSync::from_current_dir()?;
    match args {
        Args::RustcPull => {
            if let Err(error) = sync.rustc_pull(None) {
                match error {
                    RustcPullError::NothingToPull => {
                        eprintln!("Nothing to pull");
                        std::process::exit(2);
                    }
                    RustcPullError::PullFailed(error) => {
                        eprintln!("Pull failure: {error:?}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Args::RustcPush { github_username, branch } => {
            sync.rustc_push(github_username, branch)?;
        }
    }
    Ok(())
}
