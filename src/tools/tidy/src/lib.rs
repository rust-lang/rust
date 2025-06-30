//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use std::ffi::OsStr;
use std::process::Command;

use build_helper::ci::CiEnv;
use build_helper::git::{GitConfig, get_closest_upstream_commit};
use build_helper::stage0_parser::{Stage0Config, parse_stage0_file};
use termcolor::WriteColor;

macro_rules! static_regex {
    ($re:literal) => {{
        static RE: ::std::sync::OnceLock<::regex::Regex> = ::std::sync::OnceLock::new();
        RE.get_or_init(|| ::regex::Regex::new($re).unwrap())
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

macro_rules! tidy_error {
    ($bad:expr, $($fmt:tt)*) => ({
        $crate::tidy_error(&format_args!($($fmt)*).to_string()).expect("failed to output error");
        *$bad = true;
    });
}

macro_rules! tidy_error_ext {
    ($tidy_error:path, $bad:expr, $($fmt:tt)*) => ({
        $tidy_error(&format_args!($($fmt)*).to_string()).expect("failed to output error");
        *$bad = true;
    });
}

fn tidy_error(args: &str) -> std::io::Result<()> {
    use std::io::Write;

    use termcolor::{Color, ColorChoice, ColorSpec, StandardStream};

    let mut stderr = StandardStream::stdout(ColorChoice::Auto);
    stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;

    write!(&mut stderr, "tidy error")?;
    stderr.set_color(&ColorSpec::new())?;

    writeln!(&mut stderr, ": {args}")?;
    Ok(())
}

pub struct CiInfo {
    pub git_merge_commit_email: String,
    pub nightly_branch: String,
    pub base_commit: Option<String>,
    pub ci_env: CiEnv,
}

impl CiInfo {
    pub fn new(bad: &mut bool) -> Self {
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
                info.error_if_in_ci("no base commit found", bad);
                None
            }
            Err(error) => {
                info.error_if_in_ci(&format!("failed to retrieve base commit: {error}"), bad);
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

    pub fn error_if_in_ci(&self, msg: &str, bad: &mut bool) {
        if self.ci_env.is_running_in_ci() {
            *bad = true;
            eprintln!("tidy check error: {msg}");
        } else {
            eprintln!("tidy check warning: {msg}. Some checks will be skipped.");
        }
    }
}

pub fn git_diff<S: AsRef<OsStr>>(base_commit: &str, extra_arg: S) -> Option<String> {
    let output = Command::new("git").arg("diff").arg(base_commit).arg(extra_arg).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into())
}

pub mod alphabetical;
pub mod bins;
pub mod debug_artifacts;
pub mod deps;
pub mod edition;
pub mod error_codes;
pub mod ext_tool_checks;
pub mod extdeps;
pub mod features;
pub mod fluent_alphabetical;
pub mod fluent_period;
mod fluent_used;
pub mod gcc_submodule;
pub(crate) mod iter_header;
pub mod known_bug;
pub mod mir_opt_tests;
pub mod pal;
pub mod rustdoc_css_themes;
pub mod rustdoc_gui_tests;
pub mod rustdoc_js;
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
