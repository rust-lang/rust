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
//! * All unstable lang features have tests to ensure they are actually unstable

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(Debug, PartialEq, Clone)]
pub enum Status {
    Stable,
    Removed,
    Unstable,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let as_str = match *self {
            Status::Stable => "stable",
            Status::Unstable => "unstable",
            Status::Removed => "removed",
        };
        fmt::Display::fmt(as_str, f)
    }
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub level: Status,
    pub since: String,
    pub has_gate_test: bool,
    pub tracking_issue: Option<u32>,
}

impl Feature {
    fn check_match(&self, other: &Feature)-> Result<(), Vec<&'static str>> {
        let mut mismatches = Vec::new();
        if self.level != other.level {
            mismatches.push("stability level");
        }
        if self.level == Status::Stable || other.level == Status::Stable {
            // As long as a feature is unstable, the since field tracks
            // when the given part of the feature has been implemented.
            // Mismatches are tolerable as features evolve and functionality
            // gets added.
            // Once a feature is stable, the since field tracks the first version
            // it was part of the stable distribution, and mismatches are disallowed.
            if self.since != other.since {
                mismatches.push("since");
            }
        }
        if self.tracking_issue != other.tracking_issue {
            mismatches.push("tracking issue");
        }
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(mismatches)
        }
    }
}

pub type Features = HashMap<String, Feature>;

pub fn check(path: &Path, bad: &mut bool, quiet: bool) {
    let mut features = collect_lang_features(path);
    assert!(!features.is_empty());

    let lib_features = get_and_check_lib_features(path, bad, &features);
    assert!(!lib_features.is_empty());

    let mut contents = String::new();

    super::walk_many(&[&path.join("test/ui-fulldeps"),
                       &path.join("test/ui"),
                       &path.join("test/compile-fail"),
                       &path.join("test/compile-fail-fulldeps"),
                       &path.join("test/parse-fail"),
                       &path.join("test/ui"),],
                     &mut |path| super::filter_dirs(path),
                     &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        if !filename.ends_with(".rs") || filename == "features.rs" ||
           filename == "diagnostic_list.rs" {
            return;
        }

        let filen_underscore = filename.replace("-","_").replace(".rs","");
        let filename_is_gate_test = test_filen_gate(&filen_underscore, &mut features);

        contents.truncate(0);
        t!(t!(File::open(&file), &file).read_to_string(&mut contents));

        for (i, line) in contents.lines().enumerate() {
            let mut err = |msg: &str| {
                tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
            };

            let gate_test_str = "gate-test-";

            if !line.contains(gate_test_str) {
                continue;
            }

            let feature_name = match line.find(gate_test_str) {
                Some(i) => {
                    &line[i+gate_test_str.len()..line[i+1..].find(' ').unwrap_or(line.len())]
                },
                None => continue,
            };
            match features.get_mut(feature_name) {
                Some(f) => {
                    if filename_is_gate_test {
                        err(&format!("The file is already marked as gate test \
                                      through its name, no need for a \
                                      'gate-test-{}' comment",
                                     feature_name));
                    }
                    f.has_gate_test = true;
                }
                None => {
                    err(&format!("gate-test test found referencing a nonexistent feature '{}'",
                                 feature_name));
                }
            }
        }
    });

    // Only check the number of lang features.
    // Obligatory testing for library features is dumb.
    let gate_untested = features.iter()
                                .filter(|&(_, f)| f.level == Status::Unstable)
                                .filter(|&(_, f)| !f.has_gate_test)
                                .collect::<Vec<_>>();

    for &(name, _) in gate_untested.iter() {
        println!("Expected a gate test for the feature '{}'.", name);
        println!("Hint: create a failing test file named 'feature-gate-{}.rs'\
                \n      in the 'ui' test suite, with its failures due to\
                \n      missing usage of #![feature({})].", name, name);
        println!("Hint: If you already have such a test and don't want to rename it,\
                \n      you can also add a // gate-test-{} line to the test file.",
                 name);
    }

    if gate_untested.len() > 0 {
        tidy_error!(bad, "Found {} features without a gate test.", gate_untested.len());
    }

    if *bad {
        return;
    }
    if quiet {
        println!("* {} features", features.len());
        return;
    }

    let mut lines = Vec::new();
    for (name, feature) in features.iter() {
        lines.push(format!("{:<32} {:<8} {:<12} {:<8}",
                           name,
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

fn test_filen_gate(filen_underscore: &str, features: &mut Features) -> bool {
    if filen_underscore.starts_with("feature_gate") {
        for (n, f) in features.iter_mut() {
            if filen_underscore == format!("feature_gate_{}", n) {
                f.has_gate_test = true;
                return true;
            }
        }
    }
    return false;
}

pub fn collect_lang_features(base_src_path: &Path) -> Features {
    let mut contents = String::new();
    let path = base_src_path.join("libsyntax/feature_gate.rs");
    t!(t!(File::open(path)).read_to_string(&mut contents));

    contents.lines()
        .filter_map(|line| {
            let mut parts = line.trim().split(",");
            let level = match parts.next().map(|l| l.trim().trim_left_matches('(')) {
                Some("active") => Status::Unstable,
                Some("removed") => Status::Removed,
                Some("accepted") => Status::Stable,
                _ => return None,
            };
            let name = parts.next().unwrap().trim();
            let since = parts.next().unwrap().trim().trim_matches('"');
            let issue_str = parts.next().unwrap().trim();
            let tracking_issue = if issue_str.starts_with("None") {
                None
            } else {
                let s = issue_str.split("(").nth(1).unwrap().split(")").nth(0).unwrap();
                Some(s.parse().unwrap())
            };
            Some((name.to_owned(),
                Feature {
                    level,
                    since: since.to_owned(),
                    has_gate_test: false,
                    tracking_issue,
                }))
        })
        .collect()
}

pub fn collect_lib_features(base_src_path: &Path) -> Features {
    let mut lib_features = Features::new();

    // This library feature is defined in the `compiler_builtins` crate, which
    // has been moved out-of-tree. Now it can no longer be auto-discovered by
    // `tidy`, because we need to filter out its (submodule) directory. Manually
    // add it to the set of known library features so we can still generate docs.
    lib_features.insert("compiler_builtins_lib".to_owned(), Feature {
        level: Status::Unstable,
        since: "".to_owned(),
        has_gate_test: false,
        tracking_issue: None,
    });

    map_lib_features(base_src_path,
                     &mut |res, _, _| {
        match res {
            Ok((name, feature)) => {
                if lib_features.get(name).is_some() {
                    return;
                }
                lib_features.insert(name.to_owned(), feature);
            },
            Err(_) => (),
        }
    });
   lib_features
}

fn get_and_check_lib_features(base_src_path: &Path,
                              bad: &mut bool,
                              lang_features: &Features) -> Features {
    let mut lib_features = Features::new();
    map_lib_features(base_src_path,
                     &mut |res, file, line| {
            match res {
                Ok((name, f)) => {
                    let mut check_features = |f: &Feature, list: &Features, display: &str| {
                        if let Some(ref s) = list.get(name) {
                            if let Err(m) = (&f).check_match(s) {
                                tidy_error!(bad,
                                            "{}:{}: mismatches to {} in: {:?}",
                                            file.display(),
                                            line,
                                            display,
                                            &m);
                            }
                        }
                    };
                    check_features(&f, &lang_features, "corresponding lang feature");
                    check_features(&f, &lib_features, "previous");
                    lib_features.insert(name.to_owned(), f);
                },
                Err(msg) => {
                    tidy_error!(bad, "{}:{}: {}", file.display(), line, msg);
                },
            }

    });
    lib_features
}

fn map_lib_features(base_src_path: &Path,
                    mf: &mut FnMut(Result<(&str, Feature), &str>, &Path, usize)) {
    let mut contents = String::new();
    super::walk(base_src_path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
                &mut |file| {
        let filename = file.file_name().unwrap().to_string_lossy();
        if !filename.ends_with(".rs") || filename == "features.rs" ||
           filename == "diagnostic_list.rs" {
            return;
        }

        contents.truncate(0);
        t!(t!(File::open(&file), &file).read_to_string(&mut contents));

        let mut becoming_feature: Option<(String, Feature)> = None;
        for (i, line) in contents.lines().enumerate() {
            macro_rules! err {
                ($msg:expr) => {{
                    mf(Err($msg), file, i + 1);
                    continue;
                }};
            };
            if let Some((ref name, ref mut f)) = becoming_feature {
                if f.tracking_issue.is_none() {
                    f.tracking_issue = find_attr_val(line, "issue")
                    .map(|s| s.parse().unwrap());
                }
                if line.ends_with("]") {
                    mf(Ok((name, f.clone())), file, i + 1);
                } else if !line.ends_with(",") && !line.ends_with("\\") {
                    // We need to bail here because we might have missed the
                    // end of a stability attribute above because the "]"
                    // might not have been at the end of the line.
                    // We could then get into the very unfortunate situation that
                    // we continue parsing the file assuming the current stability
                    // attribute has not ended, and ignoring possible feature
                    // attributes in the process.
                    err!("malformed stability attribute");
                } else {
                    continue;
                }
            }
            becoming_feature = None;
            if line.contains("rustc_const_unstable(") {
                // const fn features are handled specially
                let feature_name = match find_attr_val(line, "feature") {
                    Some(name) => name,
                    None => err!("malformed stability attribute"),
                };
                let feature = Feature {
                    level: Status::Unstable,
                    since: "None".to_owned(),
                    has_gate_test: false,
                    // Whether there is a common tracking issue
                    // for these feature gates remains an open question
                    // https://github.com/rust-lang/rust/issues/24111#issuecomment-340283184
                    // But we take 24111 otherwise they will be shown as
                    // "internal to the compiler" which they are not.
                    tracking_issue: Some(24111),
                };
                mf(Ok((feature_name, feature)), file, i + 1);
                continue;
            }
            let level = if line.contains("[unstable(") {
                Status::Unstable
            } else if line.contains("[stable(") {
                Status::Stable
            } else {
                continue;
            };
            let feature_name = match find_attr_val(line, "feature") {
                Some(name) => name,
                None => err!("malformed stability attribute"),
            };
            let since = match find_attr_val(line, "since") {
                Some(name) => name,
                None if level == Status::Stable => {
                    err!("malformed stability attribute");
                }
                None => "None",
            };
            let tracking_issue = find_attr_val(line, "issue").map(|s| s.parse().unwrap());

            let feature = Feature {
                level,
                since: since.to_owned(),
                has_gate_test: false,
                tracking_issue,
            };
            if line.contains("]") {
                mf(Ok((feature_name, feature)), file, i + 1);
            } else {
                becoming_feature = Some((feature_name.to_owned(), feature));
            }
        }
    });
}
