use std::process::Command;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CiEnv {
    /// Not a CI environment.
    None,
    /// The Azure Pipelines environment, for Linux (including Docker), Windows, and macOS builds.
    AzurePipelines,
    /// The GitHub Actions environment, for Linux (including Docker), Windows and macOS builds.
    GitHubActions,
}

impl CiEnv {
    /// Obtains the current CI environment.
    pub fn current() -> CiEnv {
        if std::env::var("TF_BUILD").map_or(false, |e| e == "True") {
            CiEnv::AzurePipelines
        } else if std::env::var("GITHUB_ACTIONS").map_or(false, |e| e == "true") {
            CiEnv::GitHubActions
        } else {
            CiEnv::None
        }
    }

    pub fn is_ci() -> bool {
        Self::current() != CiEnv::None
    }

    /// If in a CI environment, forces the command to run with colors.
    pub fn force_coloring_in_ci(self, cmd: &mut Command) {
        if self != CiEnv::None {
            // Due to use of stamp/docker, the output stream of rustbuild is not
            // a TTY in CI, so coloring is by-default turned off.
            // The explicit `TERM=xterm` environment is needed for
            // `--color always` to actually work. This env var was lost when
            // compiling through the Makefile. Very strange.
            cmd.env("TERM", "xterm").args(&["--color", "always"]);
        }
    }
}

pub mod gha {
    /// All github actions log messages from this call to the Drop of the return value
    /// will be grouped and hidden by default in logs. Note that nesting these does
    /// not really work.
    pub fn group(name: impl std::fmt::Display) -> Group {
        if std::env::var_os("GITHUB_ACTIONS").is_some() {
            eprintln!("::group::{name}");
        } else {
            eprintln!("{name}")
        }
        Group(())
    }

    /// A guard that closes the current github actions log group on drop.
    #[must_use]
    pub struct Group(());

    impl Drop for Group {
        fn drop(&mut self) {
            if std::env::var_os("GITHUB_ACTIONS").is_some() {
                eprintln!("::endgroup::");
            }
        }
    }
}
