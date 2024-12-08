//! Checks that test revision names appearing in header directives and error
//! annotations have actually been declared in `revisions`.

// FIXME(jieyouxu) Ideally these checks would be integrated into compiletest's
// own directive and revision handling, but for now they've been split out as a
// separate `tidy` check to avoid making compiletest even messier.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use ignore::DirEntry;
use regex::Regex;

use crate::iter_header::{HeaderLine, iter_header};
use crate::walk::{filter_dirs, filter_not_rust, walk};

pub fn check(tests_path: impl AsRef<Path>, bad: &mut bool) {
    walk(
        tests_path.as_ref(),
        |path, is_dir| {
            filter_dirs(path) || filter_not_rust(path) || {
                // Auxiliary source files for incremental tests can refer to revisions
                // declared by the main file, which this check doesn't handle.
                is_dir && path.file_name().is_some_and(|name| name == "auxiliary")
            }
        },
        &mut |entry, contents| visit_test_file(entry, contents, bad),
    );
}

fn visit_test_file(entry: &DirEntry, contents: &str, bad: &mut bool) {
    let mut revisions = HashSet::new();
    let mut unused_revision_names = HashSet::new();

    // Maps each mentioned revision to the first line it was mentioned on.
    let mut mentioned_revisions = HashMap::<&str, usize>::new();
    let mut add_mentioned_revision = |line_number: usize, revision| {
        let first_line = mentioned_revisions.entry(revision).or_insert(line_number);
        *first_line = (*first_line).min(line_number);
    };

    // Scan all `//@` headers to find declared revisions and mentioned revisions.
    iter_header(contents, &mut |HeaderLine { line_number, revision, directive }| {
        if let Some(revs) = directive.strip_prefix("revisions:") {
            revisions.extend(revs.split_whitespace());
        } else if let Some(revs) = directive.strip_prefix("unused-revision-names:") {
            unused_revision_names.extend(revs.split_whitespace());
        }

        if let Some(revision) = revision {
            add_mentioned_revision(line_number, revision);
        }
    });

    // If a wildcard appears in `unused-revision-names`, skip all revision name
    // checking for this file.
    if unused_revision_names.contains(&"*") {
        return;
    }

    // Scan all `//[rev]~` error annotations to find mentioned revisions.
    for_each_error_annotation_revision(contents, &mut |ErrorAnnRev { line_number, revision }| {
        add_mentioned_revision(line_number, revision);
    });

    let path = entry.path().display();

    // Fail if any revision names appear in both places, since that's probably a mistake.
    for rev in revisions.intersection(&unused_revision_names).copied().collect::<BTreeSet<_>>() {
        tidy_error!(
            bad,
            "revision name [{rev}] appears in both `revisions` and `unused-revision-names` in {path}"
        );
    }

    // Compute the set of revisions that were mentioned but not declared,
    // sorted by the first line number they appear on.
    let mut bad_revisions = mentioned_revisions
        .into_iter()
        .filter(|(rev, _)| !revisions.contains(rev) && !unused_revision_names.contains(rev))
        .map(|(rev, line_number)| (line_number, rev))
        .collect::<Vec<_>>();
    bad_revisions.sort();

    for (line_number, rev) in bad_revisions {
        tidy_error!(bad, "unknown revision [{rev}] at {path}:{line_number}");
    }
}

struct ErrorAnnRev<'a> {
    line_number: usize,
    revision: &'a str,
}

fn for_each_error_annotation_revision<'a>(
    contents: &'a str,
    callback: &mut dyn FnMut(ErrorAnnRev<'a>),
) {
    let error_regex = {
        // Simplified from the regex used by `parse_expected` in `src/tools/compiletest/src/errors.rs`,
        // because we only care about extracting revision names.
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"//\[(?<revs>[^]]*)\]~").unwrap())
    };

    for (line_number, line) in (1..).zip(contents.lines()) {
        let Some(captures) = error_regex.captures(line) else { continue };

        for revision in captures.name("revs").unwrap().as_str().split(',') {
            callback(ErrorAnnRev { line_number, revision });
        }
    }
}
