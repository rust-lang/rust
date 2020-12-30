//! Build configuration for Rust's release channels.
//!
//! Implements the stable/beta/nightly channel distinctions by setting various
//! flags like the `unstable_features`, calculating variables like `release` and
//! `package_vers`, and otherwise indicating to the compiler what it should
//! print out as part of its version information.

use std::path::Path;
use std::process::Command;

use build_helper::output;

use crate::Build;

pub struct GitInfo {
    inner: Option<Info>,
}

struct Info {
    commit_date: String,
    sha: String,
    short_sha: String,
}

impl GitInfo {
    pub fn new(ignore_git: bool, dir: &Path) -> GitInfo {
        // See if this even begins to look like a git dir
        if ignore_git || !dir.join(".git").exists() {
            return GitInfo { inner: None };
        }

        // Make sure git commands work
        match Command::new("git").arg("rev-parse").current_dir(dir).output() {
            Ok(ref out) if out.status.success() => {}
            _ => return GitInfo { inner: None },
        }

        // Ok, let's scrape some info
        let ver_date = output(
            Command::new("git")
                .current_dir(dir)
                .arg("log")
                .arg("-1")
                .arg("--date=short")
                .arg("--pretty=format:%cd"),
        );
        let ver_hash = output(Command::new("git").current_dir(dir).arg("rev-parse").arg("HEAD"));
        let short_ver_hash = output(
            Command::new("git").current_dir(dir).arg("rev-parse").arg("--short=9").arg("HEAD"),
        );
        GitInfo {
            inner: Some(Info {
                commit_date: ver_date.trim().to_string(),
                sha: ver_hash.trim().to_string(),
                short_sha: short_ver_hash.trim().to_string(),
            }),
        }
    }

    pub fn sha(&self) -> Option<&str> {
        self.inner.as_ref().map(|s| &s.sha[..])
    }

    pub fn sha_short(&self) -> Option<&str> {
        self.inner.as_ref().map(|s| &s.short_sha[..])
    }

    pub fn commit_date(&self) -> Option<&str> {
        self.inner.as_ref().map(|s| &s.commit_date[..])
    }

    pub fn version(&self, build: &Build, num: &str) -> String {
        let mut version = build.release(num);
        if let Some(ref inner) = self.inner {
            version.push_str(" (");
            version.push_str(&inner.short_sha);
            version.push(' ');
            version.push_str(&inner.commit_date);
            version.push(')');
        }
        version
    }

    pub fn is_git(&self) -> bool {
        self.inner.is_some()
    }
}
