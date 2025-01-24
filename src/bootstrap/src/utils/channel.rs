//! Build configuration for Rust's release channels.
//!
//! Implements the stable/beta/nightly channel distinctions by setting various
//! flags like the `unstable_features`, calculating variables like `release` and
//! `package_vers`, and otherwise indicating to the compiler what it should
//! print out as part of its version information.

use std::fs;
use std::path::Path;

use super::helpers;
use crate::Build;
use crate::utils::helpers::{start_process, t};

#[derive(Clone, Default)]
pub enum GitInfo {
    /// This is not a git repository.
    #[default]
    Absent,
    /// This is a git repository.
    /// If the info should be used (`omit_git_hash` is false), this will be
    /// `Some`, otherwise it will be `None`.
    Present(Option<Info>),
    /// This is not a git repository, but the info can be fetched from the
    /// `git-commit-info` file.
    RecordedForTarball(Info),
}

#[derive(Clone)]
pub struct Info {
    pub commit_date: String,
    pub sha: String,
    pub short_sha: String,
}

impl GitInfo {
    pub fn new(omit_git_hash: bool, dir: &Path) -> GitInfo {
        // See if this even begins to look like a git dir
        if !dir.join(".git").exists() {
            match read_commit_info_file(dir) {
                Some(info) => return GitInfo::RecordedForTarball(info),
                None => return GitInfo::Absent,
            }
        }

        // Make sure git commands work
        match helpers::git(Some(dir)).arg("rev-parse").as_command_mut().output() {
            Ok(ref out) if out.status.success() => {}
            _ => return GitInfo::Absent,
        }

        // If we're ignoring the git info, we don't actually need to collect it, just make sure this
        // was a git repo in the first place.
        if omit_git_hash {
            return GitInfo::Present(None);
        }

        // Ok, let's scrape some info
        let ver_date = start_process(
            helpers::git(Some(dir))
                .arg("log")
                .arg("-1")
                .arg("--date=short")
                .arg("--pretty=format:%cd")
                .as_command_mut(),
        );
        let ver_hash =
            start_process(helpers::git(Some(dir)).arg("rev-parse").arg("HEAD").as_command_mut());
        let short_ver_hash = start_process(
            helpers::git(Some(dir)).arg("rev-parse").arg("--short=9").arg("HEAD").as_command_mut(),
        );
        GitInfo::Present(Some(Info {
            commit_date: ver_date().trim().to_string(),
            sha: ver_hash().trim().to_string(),
            short_sha: short_ver_hash().trim().to_string(),
        }))
    }

    pub fn info(&self) -> Option<&Info> {
        match self {
            GitInfo::Absent => None,
            GitInfo::Present(info) => info.as_ref(),
            GitInfo::RecordedForTarball(info) => Some(info),
        }
    }

    pub fn sha(&self) -> Option<&str> {
        self.info().map(|s| &s.sha[..])
    }

    pub fn sha_short(&self) -> Option<&str> {
        self.info().map(|s| &s.short_sha[..])
    }

    pub fn commit_date(&self) -> Option<&str> {
        self.info().map(|s| &s.commit_date[..])
    }

    pub fn version(&self, build: &Build, num: &str) -> String {
        let mut version = build.release(num);
        if let Some(inner) = self.info() {
            version.push_str(" (");
            version.push_str(&inner.short_sha);
            version.push(' ');
            version.push_str(&inner.commit_date);
            version.push(')');
        }
        version
    }

    /// Returns whether this directory has a `.git` directory which should be managed by bootstrap.
    pub fn is_managed_git_subrepository(&self) -> bool {
        match self {
            GitInfo::Absent | GitInfo::RecordedForTarball(_) => false,
            GitInfo::Present(_) => true,
        }
    }

    /// Returns whether this is being built from a tarball.
    pub fn is_from_tarball(&self) -> bool {
        match self {
            GitInfo::Absent | GitInfo::Present(_) => false,
            GitInfo::RecordedForTarball(_) => true,
        }
    }
}

/// Read the commit information from the `git-commit-info` file given the
/// project root.
pub fn read_commit_info_file(root: &Path) -> Option<Info> {
    if let Ok(contents) = fs::read_to_string(root.join("git-commit-info")) {
        let mut lines = contents.lines();
        let sha = lines.next();
        let short_sha = lines.next();
        let commit_date = lines.next();
        let info = match (commit_date, sha, short_sha) {
            (Some(commit_date), Some(sha), Some(short_sha)) => Info {
                commit_date: commit_date.to_owned(),
                sha: sha.to_owned(),
                short_sha: short_sha.to_owned(),
            },
            _ => panic!("the `git-commit-info` file is malformed"),
        };
        Some(info)
    } else {
        None
    }
}

/// Write the commit information to the `git-commit-info` file given the project
/// root.
pub fn write_commit_info_file(root: &Path, info: &Info) {
    let commit_info = format!("{}\n{}\n{}\n", info.sha, info.short_sha, info.commit_date);
    t!(fs::write(root.join("git-commit-info"), commit_info));
}

/// Write the commit hash to the `git-commit-hash` file given the project root.
pub fn write_commit_hash_file(root: &Path, sha: &str) {
    t!(fs::write(root.join("git-commit-hash"), sha));
}
