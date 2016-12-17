// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy check to ensure that unstable features are all in order
//!
//! This check will ensure properties like:
//!
//! * All stability attributes look reasonably well formed
//! * The set of library features is disjoint from the set of language features
//! * Library features have at most one stability level
//! * Library features have at most one `since` value

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(PartialEq)]
enum Status {
    Stable,
    Unstable,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let as_str = match *self {
            Status::Stable => "stable",
            Status::Unstable => "unstable",
        };
        fmt::Display::fmt(as_str, f)
    }
}


struct Feature {
    name: String,
    level: Status,
    since: String,
}

struct LibFeature {
    level: Status,
    since: String,
}

pub fn check(path: &Path, bad: &mut bool) {
    let features = collect_lang_features(&path.join("libsyntax/feature_gate.rs"));
    assert!(!features.is_empty());
    let mut lib_features = HashMap::<String, LibFeature>::new();

    let mut contents = String::new();
    super::walk(path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
                &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        if !filename.ends_with(".rs") || filename == "features.rs" ||
           filename == "diagnostic_list.rs" {
            return;
        }

        contents.truncate(0);
        t!(t!(File::open(&file), &file).read_to_string(&mut contents));

        for (i, line) in contents.lines().enumerate() {
            let mut err = |msg: &str| {
                println!("{}:{}: {}", file.display(), i + 1, msg);
                *bad = true;
            };
            let level = if line.contains("[unstable(") {
                Status::Unstable
            } else if line.contains("[stable(") {
                Status::Stable
            } else {
                continue;
            };
            let feature_name = match find_attr_val(line, "feature") {
                Some(name) => name,
                None => {
                    err("malformed stability attribute");
                    continue;
                }
            };
            let since = match find_attr_val(line, "since") {
                Some(name) => name,
                None if level == Status::Stable => {
                    err("malformed stability attribute");
                    continue;
                }
                None => "None",
            };

            if features.iter().any(|f| f.name == feature_name) {
                err("duplicating a lang feature");
            }
            if let Some(ref s) = lib_features.get(feature_name) {
                if s.level != level {
                    err("different stability level than before");
                }
                if s.since != since {
                    err("different `since` than before");
                }
                continue;
            }
            lib_features.insert(feature_name.to_owned(),
                                LibFeature {
                                    level: level,
                                    since: since.to_owned(),
                                });
        }
    });

    if *bad {
        return;
    }

    let mut lines = Vec::new();
    for feature in features {
        lines.push(format!("{:<32} {:<8} {:<12} {:<8}",
                           feature.name,
                           "lang",
                           feature.level,
                           feature.since));
    }
    for (name, feature) in lib_features {
        lines.push(format!("{:<32} {:<8} {:<12} {:<8}",
                           name,
                           "lib",
                           feature.level,
                           feature.since));
    }

    lines.sort();
    for line in lines {
        println!("* {}", line);
    }
}

fn find_attr_val<'a>(line: &'a str, attr: &str) -> Option<&'a str> {
    line.find(attr)
        .and_then(|i| line[i..].find('"').map(|j| i + j + 1))
        .and_then(|i| line[i..].find('"').map(|j| (i, i + j)))
        .map(|(i, j)| &line[i..j])
}

fn collect_lang_features(path: &Path) -> Vec<Feature> {
    let mut contents = String::new();
    t!(t!(File::open(path)).read_to_string(&mut contents));

    contents.lines()
        .filter_map(|line| {
            let mut parts = line.trim().split(",");
            let level = match parts.next().map(|l| l.trim().trim_left_matches('(')) {
                Some("active") => Status::Unstable,
                Some("removed") => Status::Unstable,
                Some("accepted") => Status::Stable,
                _ => return None,
            };
            let name = parts.next().unwrap().trim();
            let since = parts.next().unwrap().trim().trim_matches('"');
            Some(Feature {
                name: name.to_owned(),
                level: level,
                since: since.to_owned(),
            })
        })
        .collect()
}
