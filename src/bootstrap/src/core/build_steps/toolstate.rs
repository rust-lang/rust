//! [Toolstate] checks to keep tools building
//!
//! Reachable via `./x.py test` but mostly relevant for CI, since it isn't run locally by default.
//!
//! [Toolstate]: https://forge.rust-lang.org/infra/toolstate.html

use std::collections::HashMap;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::{env, fmt, fs, time};

use serde_derive::{Deserialize, Serialize};

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::utils::helpers::{self, t};

// Each cycle is 42 days long (6 weeks); the last week is 35..=42 then.
const BETA_WEEK_START: u64 = 35;

#[cfg(target_os = "linux")]
const OS: Option<&str> = Some("linux");

#[cfg(windows)]
const OS: Option<&str> = Some("windows");

#[cfg(all(not(target_os = "linux"), not(windows)))]
const OS: Option<&str> = None;

type ToolstateData = HashMap<Box<str>, ToolState>;

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
#[serde(rename_all = "kebab-case")]
/// Whether a tool can be compiled, tested or neither
pub enum ToolState {
    /// The tool compiles successfully, but the test suite fails
    TestFail = 1,
    /// The tool compiles successfully and its test suite passes
    TestPass = 2,
    /// The tool can't even be compiled
    BuildFail = 0,
}

impl fmt::Display for ToolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            ToolState::TestFail => "test-fail",
            ToolState::TestPass => "test-pass",
            ToolState::BuildFail => "build-fail",
        })
    }
}

/// Number of days after the last promotion of beta.
/// Its value is 41 on the Tuesday where "Promote master to beta (T-2)" happens.
/// The Wednesday after this has value 0.
/// We track this value to prevent regressing tools in the last week of the 6-week cycle.
fn days_since_beta_promotion() -> u64 {
    let since_epoch = t!(time::SystemTime::UNIX_EPOCH.elapsed());
    (since_epoch.as_secs() / 86400 - 20) % 42
}

// These tools must test-pass on the beta/stable channels.
//
// On the nightly channel, their build step must be attempted, but they may not
// be able to build successfully.
static STABLE_TOOLS: &[(&str, &str)] = &[
    ("book", "src/doc/book"),
    ("nomicon", "src/doc/nomicon"),
    ("reference", "src/doc/reference"),
    ("rust-by-example", "src/doc/rust-by-example"),
    ("edition-guide", "src/doc/edition-guide"),
];

// These tools are permitted to not build on the beta/stable channels.
//
// We do require that we checked whether they build or not on the tools builder,
// though, as otherwise we will be unable to file an issue if they start
// failing.
static NIGHTLY_TOOLS: &[(&str, &str)] = &[
    ("embedded-book", "src/doc/embedded-book"),
    // ("rustc-dev-guide", "src/doc/rustc-dev-guide"),
];

fn print_error(tool: &str, submodule: &str) {
    eprintln!();
    eprintln!("We detected that this PR updated '{tool}', but its tests failed.");
    eprintln!();
    eprintln!("If you do intend to update '{tool}', please check the error messages above and");
    eprintln!("commit another update.");
    eprintln!();
    eprintln!("If you do NOT intend to update '{tool}', please ensure you did not accidentally");
    eprintln!("change the submodule at '{submodule}'. You may ask your reviewer for the");
    eprintln!("proper steps.");
    crate::exit!(3);
}

fn check_changed_files(builder: &Builder<'_>, toolstates: &HashMap<Box<str>, ToolState>) {
    // Changed files
    let output = helpers::git(None)
        .arg("diff")
        .arg("--name-status")
        .arg("HEAD")
        .arg("HEAD^")
        .run_capture(builder)
        .stdout();

    for (tool, submodule) in STABLE_TOOLS.iter().chain(NIGHTLY_TOOLS.iter()) {
        let changed = output.lines().any(|l| l.starts_with('M') && l.ends_with(submodule));
        eprintln!("Verifying status of {tool}...");
        if !changed {
            continue;
        }

        eprintln!("This PR updated '{submodule}', verifying if status is 'test-pass'...");
        if toolstates[*tool] != ToolState::TestPass {
            print_error(tool, submodule);
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ToolStateCheck;

impl Step for ToolStateCheck {
    type Output = ();

    /// Checks tool state status.
    ///
    /// This is intended to be used in the `checktools.sh` script. To use
    /// this, set `save-toolstates` in `config.toml` so that tool status will
    /// be saved to a JSON file. Then, run `x.py test --no-fail-fast` for all
    /// of the tools to populate the JSON file. After that is done, this
    /// command can be run to check for any status failures, and exits with an
    /// error if there are any.
    ///
    /// This also handles publishing the results to the `history` directory of
    /// the toolstate repo <https://github.com/rust-lang-nursery/rust-toolstate>
    /// if the env var `TOOLSTATE_PUBLISH` is set. Note that there is a
    /// *separate* step of updating the `latest.json` file and creating GitHub
    /// issues and comments in `src/ci/publish_toolstate.sh`, which is only
    /// performed on master. (The shell/python code is intended to be migrated
    /// here eventually.)
    ///
    /// The rules for failure are:
    /// * If the PR modifies a tool, the status must be test-pass.
    ///   NOTE: There is intent to change this, see
    ///   <https://github.com/rust-lang/rust/issues/65000>.
    /// * All "stable" tools must be test-pass on the stable or beta branches.
    /// * During beta promotion week, a PR is not allowed to "regress" a
    ///   stable tool. That is, the status is not allowed to get worse
    ///   (test-pass to test-fail or build-fail).
    fn run(self, builder: &Builder<'_>) {
        if builder.config.dry_run() {
            return;
        }

        let days_since_beta_promotion = days_since_beta_promotion();
        let in_beta_week = days_since_beta_promotion >= BETA_WEEK_START;
        let is_nightly = !(builder.config.channel == "beta" || builder.config.channel == "stable");
        let toolstates = builder.toolstates();

        let mut did_error = false;

        for (tool, _) in STABLE_TOOLS.iter().chain(NIGHTLY_TOOLS.iter()) {
            if !toolstates.contains_key(*tool) {
                did_error = true;
                eprintln!("ERROR: Tool `{tool}` was not recorded in tool state.");
            }
        }

        if did_error {
            crate::exit!(1);
        }

        check_changed_files(builder, &toolstates);
        checkout_toolstate_repo(builder);
        let old_toolstate = read_old_toolstate();

        for (tool, _) in STABLE_TOOLS.iter() {
            let state = toolstates[*tool];

            if state != ToolState::TestPass {
                if !is_nightly {
                    did_error = true;
                    eprintln!("ERROR: Tool `{tool}` should be test-pass but is {state}");
                } else if in_beta_week {
                    let old_state = old_toolstate
                        .iter()
                        .find(|ts| ts.tool == *tool)
                        .expect("latest.json missing tool")
                        .state();
                    if state < old_state {
                        did_error = true;
                        eprintln!(
                            "ERROR: Tool `{tool}` has regressed from {old_state} to {state} during beta week."
                        );
                    } else {
                        // This warning only appears in the logs, which most
                        // people won't read. It's mostly here for testing and
                        // debugging.
                        eprintln!(
                            "WARNING: Tool `{tool}` is not test-pass (is `{state}`), \
                            this should be fixed before beta is branched."
                        );
                    }
                }
                // `publish_toolstate.py` is responsible for updating
                // `latest.json` and creating comments/issues warning people
                // if there is a regression. That all happens in a separate CI
                // job on the master branch once the PR has passed all tests
                // on the `auto` branch.
            }
        }

        if did_error {
            crate::exit!(1);
        }

        if builder.config.channel == "nightly" && env::var_os("TOOLSTATE_PUBLISH").is_some() {
            commit_toolstate_change(builder, &toolstates);
        }
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("check-tools")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ToolStateCheck);
    }
}

impl Builder<'_> {
    fn toolstates(&self) -> HashMap<Box<str>, ToolState> {
        if let Some(ref path) = self.config.save_toolstates {
            if let Some(parent) = path.parent() {
                // Ensure the parent directory always exists
                t!(std::fs::create_dir_all(parent));
            }
            let mut file = t!(fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .read(true)
                .open(path));

            serde_json::from_reader(&mut file).unwrap_or_default()
        } else {
            Default::default()
        }
    }

    /// Updates the actual toolstate of a tool.
    ///
    /// The toolstates are saved to the file specified by the key
    /// `rust.save-toolstates` in `config.toml`. If unspecified, nothing will be
    /// done. The file is updated immediately after this function completes.
    pub fn save_toolstate(&self, tool: &str, state: ToolState) {
        use std::io::Write;

        // If we're in a dry run setting we don't want to save toolstates as
        // that means if we e.g. panic down the line it'll look like we tested
        // everything (but we actually haven't).
        if self.config.dry_run() {
            return;
        }
        // Toolstate isn't tracked for clippy or rustfmt, but since most tools do, we avoid checking
        // in all the places we could save toolstate and just do so here.
        if tool == "clippy-driver" || tool == "rustfmt" {
            return;
        }
        if let Some(ref path) = self.config.save_toolstates {
            if let Some(parent) = path.parent() {
                // Ensure the parent directory always exists
                t!(std::fs::create_dir_all(parent));
            }
            let mut file = t!(fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(path));

            let mut current_toolstates: HashMap<Box<str>, ToolState> =
                serde_json::from_reader(&mut file).unwrap_or_default();
            current_toolstates.insert(tool.into(), state);
            t!(file.seek(SeekFrom::Start(0)));
            t!(file.set_len(0));
            t!(serde_json::to_writer(&file, &current_toolstates));
            t!(writeln!(file)); // make sure this ends in a newline
        }
    }
}

fn toolstate_repo() -> String {
    env::var("TOOLSTATE_REPO")
        .unwrap_or_else(|_| "https://github.com/rust-lang-nursery/rust-toolstate.git".to_string())
}

/// Directory where the toolstate repo is checked out.
const TOOLSTATE_DIR: &str = "rust-toolstate";

/// Checks out the toolstate repo into `TOOLSTATE_DIR`.
fn checkout_toolstate_repo(builder: &Builder<'_>) {
    if let Ok(token) = env::var("TOOLSTATE_REPO_ACCESS_TOKEN") {
        prepare_toolstate_config(builder, &token);
    }
    if Path::new(TOOLSTATE_DIR).exists() {
        eprintln!("Cleaning old toolstate directory...");
        t!(fs::remove_dir_all(TOOLSTATE_DIR));
    }

    helpers::git(None)
        .arg("clone")
        .arg("--depth=1")
        .arg(toolstate_repo())
        .arg(TOOLSTATE_DIR)
        .run(builder);
}

/// Sets up config and authentication for modifying the toolstate repo.
fn prepare_toolstate_config(builder: &Builder<'_>, token: &str) {
    fn git_config(builder: &Builder<'_>, key: &str, value: &str) {
        helpers::git(None).arg("config").arg("--global").arg(key).arg(value).run(builder);
    }

    // If changing anything here, then please check that `src/ci/publish_toolstate.sh` is up to date
    // as well.
    git_config(builder, "user.email", "7378925+rust-toolstate-update@users.noreply.github.com");
    git_config(builder, "user.name", "Rust Toolstate Update");
    git_config(builder, "credential.helper", "store");

    let credential = format!("https://{token}:x-oauth-basic@github.com\n",);
    let git_credential_path = PathBuf::from(t!(env::var("HOME"))).join(".git-credentials");
    t!(fs::write(git_credential_path, credential));
}

/// Reads the latest toolstate from the toolstate repo.
fn read_old_toolstate() -> Vec<RepoState> {
    let latest_path = Path::new(TOOLSTATE_DIR).join("_data").join("latest.json");
    let old_toolstate = t!(fs::read(latest_path));
    t!(serde_json::from_slice(&old_toolstate))
}

/// This function `commit_toolstate_change` provides functionality for pushing a change
/// to the `rust-toolstate` repository.
///
/// The function relies on a GitHub bot user, which should have a Personal access
/// token defined in the environment variable $TOOLSTATE_REPO_ACCESS_TOKEN. If for
/// some reason you need to change the token, please update the Azure Pipelines
/// variable group.
///
///   1. Generate a new Personal access token:
///
///       * Login to the bot account, and go to Settings -> Developer settings ->
///           Personal access tokens
///       * Click "Generate new token"
///       * Enable the "public_repo" permission, then click "Generate token"
///       * Copy the generated token (should be a 40-digit hexadecimal number).
///           Save it somewhere secure, as the token would be gone once you leave
///           the page.
///
///   2. Update the variable group in Azure Pipelines
///
///       * Ping a member of the infrastructure team to do this.
///
///   4. Replace the email address below if the bot account identity is changed
///
///       * See <https://help.github.com/articles/about-commit-email-addresses/>
///           if a private email by GitHub is wanted.
fn commit_toolstate_change(builder: &Builder<'_>, current_toolstate: &ToolstateData) {
    let message = format!("({} CI update)", OS.expect("linux/windows only"));
    let mut success = false;
    for _ in 1..=5 {
        // Upload the test results (the new commit-to-toolstate mapping) to the toolstate repo.
        // This does *not* change the "current toolstate"; that only happens post-landing
        // via `src/ci/docker/publish_toolstate.sh`.
        publish_test_results(builder, current_toolstate);

        // `git commit` failing means nothing to commit.
        let status = helpers::git(Some(Path::new(TOOLSTATE_DIR)))
            .allow_failure()
            .arg("commit")
            .arg("-a")
            .arg("-m")
            .arg(&message)
            .run(builder);
        if !status {
            success = true;
            break;
        }

        let status = helpers::git(Some(Path::new(TOOLSTATE_DIR)))
            .allow_failure()
            .arg("push")
            .arg("origin")
            .arg("master")
            .run(builder);
        // If we successfully push, exit.
        if status {
            success = true;
            break;
        }
        eprintln!("Sleeping for 3 seconds before retrying push");
        std::thread::sleep(std::time::Duration::from_secs(3));
        helpers::git(Some(Path::new(TOOLSTATE_DIR)))
            .arg("fetch")
            .arg("origin")
            .arg("master")
            .run(builder);
        helpers::git(Some(Path::new(TOOLSTATE_DIR)))
            .arg("reset")
            .arg("--hard")
            .arg("origin/master")
            .run(builder);
    }

    if !success {
        panic!("Failed to update toolstate repository with new data");
    }
}

/// Updates the "history" files with the latest results.
///
/// These results will later be promoted to `latest.json` by the
/// `publish_toolstate.py` script if the PR passes all tests and is merged to
/// master.
fn publish_test_results(builder: &Builder<'_>, current_toolstate: &ToolstateData) {
    let commit = helpers::git(None).arg("rev-parse").arg("HEAD").run_capture(builder).stdout();

    let toolstate_serialized = t!(serde_json::to_string(&current_toolstate));

    let history_path = Path::new(TOOLSTATE_DIR)
        .join("history")
        .join(format!("{}.tsv", OS.expect("linux/windows only")));
    let mut file = t!(fs::read_to_string(&history_path));
    let end_of_first_line = file.find('\n').unwrap();
    file.insert_str(end_of_first_line, &format!("\n{}\t{}", commit.trim(), toolstate_serialized));
    t!(fs::write(&history_path, file));
}

#[derive(Debug, Deserialize)]
struct RepoState {
    tool: String,
    windows: ToolState,
    linux: ToolState,
}

impl RepoState {
    fn state(&self) -> ToolState {
        if cfg!(target_os = "linux") {
            self.linux
        } else if cfg!(windows) {
            self.windows
        } else {
            unimplemented!()
        }
    }
}
