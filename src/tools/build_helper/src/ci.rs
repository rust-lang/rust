use std::process::Command;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CiEnv {
    /// Not a CI environment.
    None,
    /// The GitHub Actions environment, for Linux (including Docker), Windows and macOS builds.
    GitHubActions,
}

impl CiEnv {
    /// Obtains the current CI environment.
    pub fn current() -> CiEnv {
        if std::env::var("GITHUB_ACTIONS").map_or(false, |e| e == "true") {
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
    use std::sync::Mutex;

    static ACTIVE_GROUPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// All github actions log messages from this call to the Drop of the return value
    /// will be grouped and hidden by default in logs. Note that since github actions doesn't
    /// support group nesting, any active group will be first finished when a subgroup is started,
    /// and then re-started when the subgroup finishes.
    #[track_caller]
    pub fn group(name: impl std::fmt::Display) -> Group {
        let mut groups = ACTIVE_GROUPS.lock().unwrap();

        // A group is currently active. End it first to avoid nesting.
        if !groups.is_empty() {
            end_group();
        }

        let name = name.to_string();
        start_group(&name);
        groups.push(name);
        Group(())
    }

    /// A guard that closes the current github actions log group on drop.
    #[must_use]
    pub struct Group(());

    impl Drop for Group {
        fn drop(&mut self) {
            end_group();

            let mut groups = ACTIVE_GROUPS.lock().unwrap();
            // Remove the current group
            groups.pop();

            // If there was some previous group, restart it
            if is_in_gha() {
                if let Some(name) = groups.last() {
                    start_group(format!("{name} (continued)"));
                }
            }
        }
    }

    fn start_group(name: impl std::fmt::Display) {
        if is_in_gha() {
            eprintln!("::group::{name}");
        } else {
            eprintln!("{name}")
        }
    }

    fn end_group() {
        if is_in_gha() {
            eprintln!("::endgroup::");
        }
    }

    fn is_in_gha() -> bool {
        std::env::var_os("GITHUB_ACTIONS").is_some()
    }
}
