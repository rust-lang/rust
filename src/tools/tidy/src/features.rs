//! Tidy check to ensure that unstable features are all in order.
//!
//! This check will ensure properties like:
//!
//! * All stability attributes look reasonably well formed.
//! * The set of library features is disjoint from the set of language features.
//! * Library features have at most one stability level.
//! * Library features have at most one `since` value.
//! * All unstable lang features have tests to ensure they are actually unstable.
//! * Language features in a group are sorted by feature name.

use std::collections::hash_map::{Entry, HashMap};
use std::ffi::OsStr;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::{fmt, fs};

use crate::walk::{filter_dirs, filter_not_rust, walk, walk_many};

#[cfg(test)]
mod tests;

mod version;
use version::Version;

const FEATURE_GROUP_START_PREFIX: &str = "// feature-group-start";
const FEATURE_GROUP_END_PREFIX: &str = "// feature-group-end";

#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "build-metrics", derive(serde::Serialize))]
pub enum Status {
    Accepted,
    Removed,
    Unstable,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let as_str = match *self {
            Status::Accepted => "accepted",
            Status::Unstable => "unstable",
            Status::Removed => "removed",
        };
        fmt::Display::fmt(as_str, f)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "build-metrics", derive(serde::Serialize))]
pub struct Feature {
    pub level: Status,
    pub since: Option<Version>,
    pub has_gate_test: bool,
    pub tracking_issue: Option<NonZeroU32>,
    pub file: PathBuf,
    pub line: usize,
    pub description: Option<String>,
}
impl Feature {
    fn tracking_issue_display(&self) -> impl fmt::Display {
        match self.tracking_issue {
            None => "none".to_string(),
            Some(x) => x.to_string(),
        }
    }
}

pub type Features = HashMap<String, Feature>;

pub struct CollectedFeatures {
    pub lib: Features,
    pub lang: Features,
}

// Currently only used for unstable book generation
pub fn collect_lib_features(base_src_path: &Path) -> Features {
    let mut lib_features = Features::new();

    map_lib_features(base_src_path, &mut |res, _, _| {
        if let Ok((name, feature)) = res {
            lib_features.insert(name.to_owned(), feature);
        }
    });
    lib_features
}

pub fn check(
    src_path: &Path,
    tests_path: &Path,
    compiler_path: &Path,
    lib_path: &Path,
    bad: &mut bool,
    verbose: bool,
) -> CollectedFeatures {
    let mut features = collect_lang_features(compiler_path, bad);
    assert!(!features.is_empty());

    let lib_features = get_and_check_lib_features(lib_path, bad, &features);
    assert!(!lib_features.is_empty());

    walk_many(
        &[
            &tests_path.join("ui"),
            &tests_path.join("ui-fulldeps"),
            &tests_path.join("rustdoc-ui"),
            &tests_path.join("rustdoc"),
        ],
        |path, _is_dir| {
            filter_dirs(path)
                || filter_not_rust(path)
                || path.file_name() == Some(OsStr::new("features.rs"))
                || path.file_name() == Some(OsStr::new("diagnostic_list.rs"))
        },
        &mut |entry, contents| {
            let file = entry.path();
            let filename = file.file_name().unwrap().to_string_lossy();
            let filen_underscore = filename.replace('-', "_").replace(".rs", "");
            let filename_gate = test_filen_gate(&filen_underscore, &mut features);

            for (i, line) in contents.lines().enumerate() {
                let mut err = |msg: &str| {
                    tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
                };

                let gate_test_str = "gate-test-";

                let feature_name = match line.find(gate_test_str) {
                    // NB: the `splitn` always succeeds, even if the delimiter is not present.
                    Some(i) => line[i + gate_test_str.len()..].splitn(2, ' ').next().unwrap(),
                    None => continue,
                };
                match features.get_mut(feature_name) {
                    Some(f) => {
                        if filename_gate == Some(feature_name) {
                            err(&format!(
                                "The file is already marked as gate test \
                                      through its name, no need for a \
                                      'gate-test-{}' comment",
                                feature_name
                            ));
                        }
                        f.has_gate_test = true;
                    }
                    None => {
                        err(&format!(
                            "gate-test test found referencing a nonexistent feature '{}'",
                            feature_name
                        ));
                    }
                }
            }
        },
    );

    // Only check the number of lang features.
    // Obligatory testing for library features is dumb.
    let gate_untested = features
        .iter()
        .filter(|&(_, f)| f.level == Status::Unstable)
        .filter(|&(_, f)| !f.has_gate_test)
        .collect::<Vec<_>>();

    for &(name, _) in gate_untested.iter() {
        println!("Expected a gate test for the feature '{name}'.");
        println!(
            "Hint: create a failing test file named 'tests/ui/feature-gates/feature-gate-{}.rs',\
                \n      with its failures due to missing usage of `#![feature({})]`.",
            name.replace("_", "-"),
            name
        );
        println!(
            "Hint: If you already have such a test and don't want to rename it,\
                \n      you can also add a // gate-test-{} line to the test file.",
            name
        );
    }

    if !gate_untested.is_empty() {
        tidy_error!(bad, "Found {} features without a gate test.", gate_untested.len());
    }

    let (version, channel) = get_version_and_channel(src_path);

    let all_features_iter = features
        .iter()
        .map(|feat| (feat, "lang"))
        .chain(lib_features.iter().map(|feat| (feat, "lib")));
    for ((feature_name, feature), kind) in all_features_iter {
        let since = if let Some(since) = feature.since { since } else { continue };
        let file = feature.file.display();
        let line = feature.line;
        if since > version && since != Version::CurrentPlaceholder {
            tidy_error!(
                bad,
                "{file}:{line}: The stabilization version {since} of {kind} feature `{feature_name}` is newer than the current {version}"
            );
        }
        if channel == "nightly" && since == version {
            tidy_error!(
                bad,
                "{file}:{line}: The stabilization version {since} of {kind} feature `{feature_name}` is written out but should be {}",
                version::VERSION_PLACEHOLDER
            );
        }
        if channel != "nightly" && since == Version::CurrentPlaceholder {
            tidy_error!(
                bad,
                "{file}:{line}: The placeholder use of {kind} feature `{feature_name}` is not allowed on the {channel} channel",
            );
        }
    }

    if *bad {
        return CollectedFeatures { lib: lib_features, lang: features };
    }

    if verbose {
        let mut lines = Vec::new();
        lines.extend(format_features(&features, "lang"));
        lines.extend(format_features(&lib_features, "lib"));

        lines.sort();
        for line in lines {
            println!("* {line}");
        }
    }

    CollectedFeatures { lib: lib_features, lang: features }
}

fn get_version_and_channel(src_path: &Path) -> (Version, String) {
    let version_str = t!(std::fs::read_to_string(src_path.join("version")));
    let version_str = version_str.trim();
    let version = t!(std::str::FromStr::from_str(&version_str).map_err(|e| format!("{e:?}")));
    let channel_str = t!(std::fs::read_to_string(src_path.join("ci").join("channel")));
    (version, channel_str.trim().to_owned())
}

fn format_features<'a>(
    features: &'a Features,
    family: &'a str,
) -> impl Iterator<Item = String> + 'a {
    features.iter().map(move |(name, feature)| {
        format!(
            "{:<32} {:<8} {:<12} {:<8}",
            name,
            family,
            feature.level,
            feature.since.map_or("None".to_owned(), |since| since.to_string())
        )
    })
}

fn find_attr_val<'a>(line: &'a str, attr: &str) -> Option<&'a str> {
    let r = match attr {
        "issue" => static_regex!(r#"issue\s*=\s*"([^"]*)""#),
        "feature" => static_regex!(r#"feature\s*=\s*"([^"]*)""#),
        "since" => static_regex!(r#"since\s*=\s*"([^"]*)""#),
        _ => unimplemented!("{attr} not handled"),
    };

    r.captures(line).and_then(|c| c.get(1)).map(|m| m.as_str())
}

fn test_filen_gate<'f>(filen_underscore: &'f str, features: &mut Features) -> Option<&'f str> {
    let prefix = "feature_gate_";
    if let Some(suffix) = filen_underscore.strip_prefix(prefix) {
        for (n, f) in features.iter_mut() {
            // Equivalent to filen_underscore == format!("feature_gate_{n}")
            if suffix == n {
                f.has_gate_test = true;
                return Some(suffix);
            }
        }
    }
    None
}

pub fn collect_lang_features(base_compiler_path: &Path, bad: &mut bool) -> Features {
    let mut features = Features::new();
    collect_lang_features_in(&mut features, base_compiler_path, "accepted.rs", bad);
    collect_lang_features_in(&mut features, base_compiler_path, "removed.rs", bad);
    collect_lang_features_in(&mut features, base_compiler_path, "unstable.rs", bad);
    features
}

fn collect_lang_features_in(features: &mut Features, base: &Path, file: &str, bad: &mut bool) {
    let path = base.join("rustc_feature").join("src").join(file);
    let contents = t!(fs::read_to_string(&path));

    // We allow rustc-internal features to omit a tracking issue.
    // To make tidy accept omitting a tracking issue, group the list of features
    // without one inside `// no-tracking-issue` and `// no-tracking-issue-end`.
    let mut next_feature_omits_tracking_issue = false;

    let mut in_feature_group = false;
    let mut prev_names = vec![];

    let lines = contents.lines().zip(1..);
    let mut doc_comments: Vec<String> = Vec::new();
    for (line, line_number) in lines {
        let line = line.trim();

        // Within -start and -end, the tracking issue can be omitted.
        match line {
            "// no-tracking-issue-start" => {
                next_feature_omits_tracking_issue = true;
                continue;
            }
            "// no-tracking-issue-end" => {
                next_feature_omits_tracking_issue = false;
                continue;
            }
            _ => {}
        }

        if line.starts_with(FEATURE_GROUP_START_PREFIX) {
            if in_feature_group {
                tidy_error!(
                    bad,
                    "{}:{}: \
                        new feature group is started without ending the previous one",
                    path.display(),
                    line_number,
                );
            }

            in_feature_group = true;
            prev_names = vec![];
            continue;
        } else if line.starts_with(FEATURE_GROUP_END_PREFIX) {
            in_feature_group = false;
            prev_names = vec![];
            continue;
        }

        if in_feature_group {
            if let Some(doc_comment) = line.strip_prefix("///") {
                doc_comments.push(doc_comment.trim().to_string());
                continue;
            }
        }

        let mut parts = line.split(',');
        let level = match parts.next().map(|l| l.trim().trim_start_matches('(')) {
            Some("unstable") => Status::Unstable,
            Some("incomplete") => Status::Unstable,
            Some("internal") => Status::Unstable,
            Some("removed") => Status::Removed,
            Some("accepted") => Status::Accepted,
            _ => continue,
        };
        let name = parts.next().unwrap().trim();

        let since_str = parts.next().unwrap().trim().trim_matches('"');
        let since = match since_str.parse() {
            Ok(since) => Some(since),
            Err(err) => {
                tidy_error!(
                    bad,
                    "{}:{}: failed to parse since: {} ({:?})",
                    path.display(),
                    line_number,
                    since_str,
                    err,
                );
                None
            }
        };
        if in_feature_group {
            if prev_names.last() > Some(&name) {
                // This assumes the user adds the feature name at the end of the list, as we're
                // not looking ahead.
                let correct_index = match prev_names.binary_search(&name) {
                    Ok(_) => {
                        // This only occurs when the feature name has already been declared.
                        tidy_error!(
                            bad,
                            "{}:{}: duplicate feature {}",
                            path.display(),
                            line_number,
                            name,
                        );
                        // skip any additional checks for this line
                        continue;
                    }
                    Err(index) => index,
                };

                let correct_placement = if correct_index == 0 {
                    "at the beginning of the feature group".to_owned()
                } else if correct_index == prev_names.len() {
                    // I don't believe this is reachable given the above assumption, but it
                    // doesn't hurt to be safe.
                    "at the end of the feature group".to_owned()
                } else {
                    format!(
                        "between {} and {}",
                        prev_names[correct_index - 1],
                        prev_names[correct_index],
                    )
                };

                tidy_error!(
                    bad,
                    "{}:{}: feature {} is not sorted by feature name (should be {})",
                    path.display(),
                    line_number,
                    name,
                    correct_placement,
                );
            }
            prev_names.push(name);
        }

        let issue_str = parts.next().unwrap().trim();
        let tracking_issue = if issue_str.starts_with("None") {
            if level == Status::Unstable && !next_feature_omits_tracking_issue {
                tidy_error!(
                    bad,
                    "{}:{}: no tracking issue for feature {}",
                    path.display(),
                    line_number,
                    name,
                );
            }
            None
        } else {
            let s = issue_str.split('(').nth(1).unwrap().split(')').next().unwrap();
            Some(s.parse().unwrap())
        };
        match features.entry(name.to_owned()) {
            Entry::Occupied(e) => {
                tidy_error!(
                    bad,
                    "{}:{} feature {name} already specified with status '{}'",
                    path.display(),
                    line_number,
                    e.get().level,
                );
            }
            Entry::Vacant(e) => {
                e.insert(Feature {
                    level,
                    since,
                    has_gate_test: false,
                    tracking_issue,
                    file: path.to_path_buf(),
                    line: line_number,
                    description: if doc_comments.is_empty() {
                        None
                    } else {
                        Some(doc_comments.join(" "))
                    },
                });
            }
        }
        doc_comments.clear();
    }
}

fn get_and_check_lib_features(
    base_src_path: &Path,
    bad: &mut bool,
    lang_features: &Features,
) -> Features {
    let mut lib_features = Features::new();
    map_lib_features(base_src_path, &mut |res, file, line| match res {
        Ok((name, f)) => {
            let mut check_features = |f: &Feature, list: &Features, display: &str| {
                if let Some(ref s) = list.get(name) {
                    if f.tracking_issue != s.tracking_issue && f.level != Status::Accepted {
                        tidy_error!(
                            bad,
                            "{}:{}: feature gate {} has inconsistent `issue`: \"{}\" mismatches the {} `issue` of \"{}\"",
                            file.display(),
                            line,
                            name,
                            f.tracking_issue_display(),
                            display,
                            s.tracking_issue_display(),
                        );
                    }
                }
            };
            check_features(&f, &lang_features, "corresponding lang feature");
            check_features(&f, &lib_features, "previous");
            lib_features.insert(name.to_owned(), f);
        }
        Err(msg) => {
            tidy_error!(bad, "{}:{}: {}", file.display(), line, msg);
        }
    });
    lib_features
}

fn map_lib_features(
    base_src_path: &Path,
    mf: &mut (dyn Send + Sync + FnMut(Result<(&str, Feature), &str>, &Path, usize)),
) {
    walk(
        base_src_path,
        |path, _is_dir| filter_dirs(path) || path.ends_with("tests"),
        &mut |entry, contents| {
            let file = entry.path();
            let filename = file.file_name().unwrap().to_string_lossy();
            if !filename.ends_with(".rs")
                || filename == "features.rs"
                || filename == "diagnostic_list.rs"
                || filename == "error_codes.rs"
            {
                return;
            }

            // This is an early exit -- all the attributes we're concerned with must contain this:
            // * rustc_const_unstable(
            // * unstable(
            // * stable(
            if !contents.contains("stable(") {
                return;
            }

            let handle_issue_none = |s| match s {
                "none" => None,
                issue => {
                    let n = issue.parse().expect("issue number is not a valid integer");
                    assert_ne!(n, 0, "\"none\" should be used when there is no issue, not \"0\"");
                    NonZeroU32::new(n)
                }
            };
            let mut becoming_feature: Option<(&str, Feature)> = None;
            let mut iter_lines = contents.lines().enumerate().peekable();
            while let Some((i, line)) = iter_lines.next() {
                macro_rules! err {
                    ($msg:expr) => {{
                        mf(Err($msg), file, i + 1);
                        continue;
                    }};
                }

                // exclude commented out lines
                if static_regex!(r"^\s*//").is_match(line) {
                    continue;
                }

                if let Some((ref name, ref mut f)) = becoming_feature {
                    if f.tracking_issue.is_none() {
                        f.tracking_issue = find_attr_val(line, "issue").and_then(handle_issue_none);
                    }
                    if line.ends_with(']') {
                        mf(Ok((name, f.clone())), file, i + 1);
                    } else if !line.ends_with(',') && !line.ends_with('\\') && !line.ends_with('"')
                    {
                        // We need to bail here because we might have missed the
                        // end of a stability attribute above because the ']'
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
                    // `const fn` features are handled specially.
                    let feature_name = match find_attr_val(line, "feature").or_else(|| {
                        iter_lines.peek().and_then(|next| find_attr_val(next.1, "feature"))
                    }) {
                        Some(name) => name,
                        None => err!("malformed stability attribute: missing `feature` key"),
                    };
                    let feature = Feature {
                        level: Status::Unstable,
                        since: None,
                        has_gate_test: false,
                        tracking_issue: find_attr_val(line, "issue").and_then(handle_issue_none),
                        file: file.to_path_buf(),
                        line: i + 1,
                        description: None,
                    };
                    mf(Ok((feature_name, feature)), file, i + 1);
                    continue;
                }
                let level = if line.contains("[unstable(") {
                    Status::Unstable
                } else if line.contains("[stable(") {
                    Status::Accepted
                } else {
                    continue;
                };
                let feature_name = match find_attr_val(line, "feature")
                    .or_else(|| iter_lines.peek().and_then(|next| find_attr_val(next.1, "feature")))
                {
                    Some(name) => name,
                    None => err!("malformed stability attribute: missing `feature` key"),
                };
                let since = match find_attr_val(line, "since").map(|x| x.parse()) {
                    Some(Ok(since)) => Some(since),
                    Some(Err(_err)) => {
                        err!("malformed stability attribute: can't parse `since` key");
                    }
                    None if level == Status::Accepted => {
                        err!("malformed stability attribute: missing the `since` key");
                    }
                    None => None,
                };
                let tracking_issue = find_attr_val(line, "issue").and_then(handle_issue_none);

                let feature = Feature {
                    level,
                    since,
                    has_gate_test: false,
                    tracking_issue,
                    file: file.to_path_buf(),
                    line: i + 1,
                    description: None,
                };
                if line.contains(']') {
                    mf(Ok((feature_name, feature)), file, i + 1);
                } else {
                    becoming_feature = Some((feature_name, feature));
                }
            }
        },
    );
}
