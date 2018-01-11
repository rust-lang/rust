// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Build configuration for Rust's release channels.
//!
//! Implements the stable/beta/nightly channel distinctions by setting various
//! flags like the `unstable_features`, calculating variables like `release` and
//! `package_vers`, and otherwise indicating to the compiler what it should
//! print out as part of its version information.

use std::path::Path;
use std::process::Command;

use build_helper::output;

use Build;
use config::Config;

// The version number
pub const CFG_RELEASE_NUM: &str = "1.25.0";

// An optional number to put after the label, e.g. '.2' -> '-beta.2'
// Be sure to make this starts with a dot to conform to semver pre-release
// versions (section 9)
pub const CFG_PRERELEASE_VERSION: &str = ".1";

pub struct GitInfo {
    inner: Option<Info>,
}

struct Info {
    commit_date: String,
    sha: String,
    short_sha: String,
}

impl GitInfo {
    pub fn new(config: &Config, dir: &Path) -> GitInfo {
        // See if this even begins to look like a git dir
        if config.ignore_git || !dir.join(".git").exists() {
            return GitInfo { inner: None }
        }

        // Make sure git commands work
        let out = Command::new("git")
                          .arg("rev-parse")
                          .current_dir(dir)
                          .output()
                          .expect("failed to spawn git");
        if !out.status.success() {
            return GitInfo { inner: None }
        }

        // Ok, let's scrape some info
        let ver_date = output(Command::new("git").current_dir(dir)
                                      .arg("log").arg("-1")
                                      .arg("--date=short")
                                      .arg("--pretty=format:%cd"));
        let ver_hash = output(Command::new("git").current_dir(dir)
                                      .arg("rev-parse").arg("HEAD"));
        let short_ver_hash = output(Command::new("git")
                                            .current_dir(dir)
                                            .arg("rev-parse")
                                            .arg("--short=9")
                                            .arg("HEAD"));
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
            version.push_str(" ");
            version.push_str(&inner.commit_date);
            version.push_str(")");
        }
        version
    }

    pub fn is_git(&self) -> bool {
        self.inner.is_some()
    }
}
