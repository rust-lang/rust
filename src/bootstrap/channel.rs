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

use std::process::Command;

use build_helper::output;

use Build;

// The version number
const CFG_RELEASE_NUM: &'static str = "1.17.0";

// An optional number to put after the label, e.g. '.2' -> '-beta.2'
// Be sure to make this starts with a dot to conform to semver pre-release
// versions (section 9)
const CFG_PRERELEASE_VERSION: &'static str = ".1";

pub fn collect(build: &mut Build) {
    build.release_num = CFG_RELEASE_NUM.to_string();
    build.prerelease_version = CFG_RELEASE_NUM.to_string();

    // Depending on the channel, passed in `./configure --release-channel`,
    // determine various properties of the build.
    match &build.config.channel[..] {
        "stable" => {
            build.release = CFG_RELEASE_NUM.to_string();
            build.package_vers = build.release.clone();
            build.unstable_features = false;
        }
        "beta" => {
            build.release = format!("{}-beta{}", CFG_RELEASE_NUM,
                                   CFG_PRERELEASE_VERSION);
            build.package_vers = "beta".to_string();
            build.unstable_features = false;
        }
        "nightly" => {
            build.release = format!("{}-nightly", CFG_RELEASE_NUM);
            build.package_vers = "nightly".to_string();
            build.unstable_features = true;
        }
        _ => {
            build.release = format!("{}-dev", CFG_RELEASE_NUM);
            build.package_vers = build.release.clone();
            build.unstable_features = true;
        }
    }
    build.version = build.release.clone();

    // If we have a git directory, add in some various SHA information of what
    // commit this compiler was compiled from.
    if build.src.join(".git").is_dir() {
        let ver_date = output(Command::new("git").current_dir(&build.src)
                                      .arg("log").arg("-1")
                                      .arg("--date=short")
                                      .arg("--pretty=format:%cd"));
        let ver_hash = output(Command::new("git").current_dir(&build.src)
                                      .arg("rev-parse").arg("HEAD"));
        let short_ver_hash = output(Command::new("git")
                                            .current_dir(&build.src)
                                            .arg("rev-parse")
                                            .arg("--short=9")
                                            .arg("HEAD"));
        let ver_date = ver_date.trim().to_string();
        let ver_hash = ver_hash.trim().to_string();
        let short_ver_hash = short_ver_hash.trim().to_string();
        build.version.push_str(&format!(" ({} {})", short_ver_hash,
                                       ver_date));
        build.ver_date = Some(ver_date.to_string());
        build.ver_hash = Some(ver_hash);
        build.short_ver_hash = Some(short_ver_hash);
    }
}
