// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::process::Command;

use build_helper::output;

use build::Build;
use build::util::mtime;

pub fn collect(build: &mut Build) {
    let mut main_mk = String::new();
    t!(t!(File::open(build.src.join("mk/main.mk"))).read_to_string(&mut main_mk));
    let mut release_num = "";
    let mut prerelease_version = "";
    for line in main_mk.lines() {
        if line.starts_with("CFG_RELEASE_NUM") {
            release_num = line.split('=').skip(1).next().unwrap().trim();
        }
        if line.starts_with("CFG_PRERELEASE_VERSION") {
            prerelease_version = line.split('=').skip(1).next().unwrap().trim();
        }
    }

    // FIXME: this is duplicating makefile logic
    match &build.config.channel[..] {
        "stable" => {
            build.release = release_num.to_string();
            build.unstable_features = false;
        }
        "beta" => {
            build.release = format!("{}-beta{}", release_num,
                                   prerelease_version);
            build.unstable_features = false;
        }
        "nightly" => {
            build.release = format!("{}-nightly", release_num);
            build.unstable_features = true;
        }
        _ => {
            build.release = format!("{}-dev", release_num);
            build.unstable_features = true;
        }
    }
    build.version = build.release.clone();

    if fs::metadata(build.src.join(".git")).is_ok() {
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

    build.bootstrap_key = mtime(Path::new("config.toml")).seconds()
                                                        .to_string();
}

