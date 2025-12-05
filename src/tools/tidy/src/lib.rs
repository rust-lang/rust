//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, io};

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

/// If the given executable is installed with the given version, use that,
/// otherwise install via cargo.
pub fn ensure_version_or_cargo_install(
    build_dir: &Path,
    cargo: &Path,
    pkg_name: &str,
    bin_name: &str,
    version: &str,
) -> io::Result<PathBuf> {
    let tool_root_dir = build_dir.join("misc-tools");
    let tool_bin_dir = tool_root_dir.join("bin");
    let bin_path = tool_bin_dir.join(bin_name).with_extension(env::consts::EXE_EXTENSION);

    // ignore the process exit code here and instead just let the version number check fail.
    // we also importantly don't return if the program wasn't installed,
    // instead we want to continue to the fallback.
    'ck: {
        // FIXME: rewrite as if-let chain once this crate is 2024 edition.
        let Ok(output) = Command::new(&bin_path).arg("--version").output() else {
            break 'ck;
        };
        let Ok(s) = str::from_utf8(&output.stdout) else {
            break 'ck;
        };
        let Some(v) = s.trim().split_whitespace().last() else {
            break 'ck;
        };
        if v == version {
            return Ok(bin_path);
        }
    }

    eprintln!("building external tool {bin_name} from package {pkg_name}@{version}");
    // use --force to ensure that if the required version is bumped, we update it.
    // use --target-dir to ensure we have a build cache so repeated invocations aren't slow.
    // modify PATH so that cargo doesn't print a warning telling the user to modify the path.
    let mut cmd = Command::new(cargo);
    cmd.args(["install", "--locked", "--force", "--quiet"])
        .arg("--root")
        .arg(&tool_root_dir)
        .arg("--target-dir")
        .arg(tool_root_dir.join("target"))
        .arg(format!("{pkg_name}@{version}"))
        .env(
            "PATH",
            env::join_paths(
                env::split_paths(&env::var("PATH").unwrap())
                    .chain(std::iter::once(tool_bin_dir.clone())),
            )
            .expect("build dir contains invalid char"),
        );

    // On CI, we set opt-level flag for quicker installation.
    // Since lower opt-level decreases the tool's performance,
    // we don't set this option on local.
    if CiEnv::is_ci() {
        cmd.env("RUSTFLAGS", "-Copt-level=0");
    }

    let cargo_exit_code = cmd.spawn()?.wait()?;
    if !cargo_exit_code.success() {
        return Err(io::Error::other("cargo install failed"));
    }
    assert!(
        matches!(bin_path.try_exists(), Ok(true)),
        "cargo install did not produce the expected binary"
    );
    eprintln!("finished building tool {bin_name}");
    Ok(bin_path)
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
