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
        Self::current().is_running_in_ci()
    }

    pub fn is_running_in_ci(self) -> bool {
        self != CiEnv::None
    }

    /// Checks if running in rust-lang/rust managed CI job.
    pub fn is_rust_lang_managed_ci_job() -> bool {
        Self::is_ci()
            // If both are present, we can assume it's an upstream CI job
            // as they are always set unconditionally.
            && std::env::var_os("CI_JOB_NAME").is_some()
            && std::env::var_os("TOOLSTATE_REPO").is_some()
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
            println!("::group::{name}");
        } else {
            println!("{name}")
        }
    }

    fn end_group() {
        if is_in_gha() {
            println!("::endgroup::");
        }
    }

    fn is_in_gha() -> bool {
        std::env::var_os("GITHUB_ACTIONS").is_some()
    }
}
