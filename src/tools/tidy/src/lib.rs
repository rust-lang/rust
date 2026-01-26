//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use std::ffi::OsStr;
use std::process::Command;

use build_helper::ci::CiEnv;
use build_helper::git::{GitConfig, get_closest_upstream_commit};
use build_helper::stage0_parser::{Stage0Config, parse_stage0_file};

use crate::diagnostics::{RunningCheck, TidyCtx};

macro_rules! static_regex {
    ($re:literal) => {{
        static RE: ::std::sync::LazyLock<::regex::Regex> =
            ::std::sync::LazyLock::new(|| ::regex::Regex::new($re).unwrap());
        &*RE
    }};
}

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The expression that failed
/// * The error itself
/// * (optionally) a path connected to the error (e.g. failure to open a file)
#[macro_export]
macro_rules! t {
    ($e:expr, $p:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed on {} with {}", stringify!($e), ($p).display(), e),
        }
    };

    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
}

pub struct CiInfo {
    pub git_merge_commit_email: String,
    pub nightly_branch: String,
    pub base_commit: Option<String>,
    pub ci_env: CiEnv,
}

impl CiInfo {
    pub fn new(tidy_ctx: TidyCtx) -> Self {
        let mut check = tidy_ctx.start_check("CI history");

        let stage0 = parse_stage0_file();
        let Stage0Config { nightly_branch, git_merge_commit_email, .. } = stage0.config;

        let mut info = Self {
            nightly_branch,
            git_merge_commit_email,
            ci_env: CiEnv::current(),
            base_commit: None,
        };
        let base_commit = match get_closest_upstream_commit(None, &info.git_config(), info.ci_env) {
            Ok(Some(commit)) => Some(commit),
            Ok(None) => {
                info.error_if_in_ci("no base commit found", &mut check);
                None
            }
            Err(error) => {
                info.error_if_in_ci(
                    &format!("failed to retrieve base commit: {error}"),
                    &mut check,
                );
                None
            }
        };
        info.base_commit = base_commit;
        info
    }

    pub fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            nightly_branch: &self.nightly_branch,
            git_merge_commit_email: &self.git_merge_commit_email,
        }
    }

    pub fn error_if_in_ci(&self, msg: &str, check: &mut RunningCheck) {
        if self.ci_env.is_running_in_ci() {
            check.error(msg);
        } else {
            check.warning(format!("{msg}. Some checks will be skipped."));
        }
    }
}

pub fn git_diff<S: AsRef<OsStr>>(base_commit: &str, extra_arg: S) -> Option<String> {
    let output = Command::new("git").arg("diff").arg(base_commit).arg(extra_arg).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into())
}

/// Similar to `files_modified`, but only involves a single call to `git`.
///
/// removes all elements from `items` that do not cause any match when `pred` is called with the list of modifed files.
///
/// if in CI, no elements will be removed.
pub fn files_modified_batch_filter<T>(
    ci_info: &CiInfo,
    items: &mut Vec<T>,
    pred: impl Fn(&T, &str) -> bool,
) {
    if CiEnv::is_ci() {
        // assume everything is modified on CI because we really don't want false positives there.
        return;
    }
    let Some(base_commit) = &ci_info.base_commit else {
        eprintln!("No base commit, assuming all files are modified");
        return;
    };
    match crate::git_diff(base_commit, "--name-status") {
        Some(output) => {
            let modified_files: Vec<_> = output
                .lines()
                .filter_map(|ln| {
                    let (status, name) = ln
                        .trim_end()
                        .split_once('\t')
                        .expect("bad format from `git diff --name-status`");
                    if status == "M" { Some(name) } else { None }
                })
                .collect();
            items.retain(|item| {
                for modified_file in &modified_files {
                    if pred(item, modified_file) {
                        // at least one predicate matches, keep this item.
                        return true;
                    }
                }
                // no predicates matched, remove this item.
                false
            });
        }
        None => {
            eprintln!("warning: failed to run `git diff` to check for changes");
            eprintln!("warning: assuming all files are modified");
        }
    }
}

/// Returns true if any modified file matches the predicate, if we are in CI, or if unable to list modified files.
pub fn files_modified(ci_info: &CiInfo, pred: impl Fn(&str) -> bool) -> bool {
    let mut v = vec![()];
    files_modified_batch_filter(ci_info, &mut v, |_, p| pred(p));
    !v.is_empty()
}

pub mod alphabetical;
pub mod bins;
pub mod debug_artifacts;
pub mod deps;
pub mod diagnostics;
pub mod edition;
pub mod error_codes;
pub mod extdeps;
pub mod extra_checks;
pub mod features;
pub mod filenames;
pub mod fluent_alphabetical;
pub mod fluent_lowercase;
pub mod fluent_period;
mod fluent_used;
pub mod gcc_submodule;
pub(crate) mod iter_header;
pub mod known_bug;
pub mod mir_opt_tests;
pub mod pal;
pub mod rustdoc_css_themes;
pub mod rustdoc_gui_tests;
pub mod rustdoc_json;
pub mod rustdoc_templates;
pub mod style;
pub mod target_policy;
pub mod target_specific_tests;
pub mod tests_placement;
pub mod tests_revision_unpaired_stdout_stderr;
pub mod triagebot;
pub mod ui_tests;
pub mod unit_tests;
pub mod unknown_revision;
pub mod unstable_book;
pub mod walk;
pub mod x_version;
