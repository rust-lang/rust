//! This checks the output of `--print=crate-root-lint-levels`

use std::collections::HashSet;
use std::iter::FromIterator;

use run_make_support::rustdoc;

struct CrateRootLintLevels {
    args: &'static [&'static str],
    contains: Contains,
}

struct Contains {
    contains: &'static [&'static str],
    doesnt_contain: &'static [&'static str],
}

fn main() {
    // rustdoc don't run rustc lints, and ignores rustc lint check attributes
    check(CrateRootLintLevels {
        args: &[],
        contains: Contains {
            contains: &[
                "rustdoc::private_doc_tests=allow",
                "unused_mut=allow",
                "warnings=warn",
                "stable_features=warn",
                "unknown_lints=warn",
                "rustdoc::broken_intra_doc_links=warn",
                "rustdoc::private_intra_doc_links=forbid",
                "rustdoc::missing_crate_level_docs=allow",
            ],
            doesnt_contain: &["rustdoc::private_doc_tests=warn", "unused_mut=expect"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Wrustdoc::private_doc_tests"],
        contains: Contains {
            contains: &["rustdoc::private_doc_tests=allow", "warnings=warn"],
            doesnt_contain: &["rustdoc::private_doc_tests=warn"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Dwarnings"],
        contains: Contains {
            contains: &[
                "rustdoc::private_doc_tests=allow",
                "warnings=deny",
                "stable_features=deny",
                "unknown_lints=deny",
            ],
            doesnt_contain: &["warnings=warn"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Dstable_features"],
        contains: Contains {
            contains: &[
                "warnings=warn",
                "stable_features=deny",
                "rustdoc::private_doc_tests=allow",
            ],
            doesnt_contain: &["warnings=deny"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Dwarnings", "--force-warn=stable_features"],
        contains: Contains {
            contains: &["warnings=deny", "stable_features=force-warn", "unknown_lints=deny"],
            doesnt_contain: &["warnings=warn"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Dwarnings", "--cap-lints=warn"],
        contains: Contains {
            contains: &[
                "rustdoc::private_doc_tests=allow",
                "warnings=warn",
                "stable_features=warn",
                "unknown_lints=warn",
            ],
            doesnt_contain: &["warnings=deny"],
        },
    });
}

#[track_caller]
fn check(CrateRootLintLevels { args, contains }: CrateRootLintLevels) {
    let output = rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .arg("--print=crate-root-lint-levels")
        .args(args)
        .run();

    let stdout = output.stdout_utf8();

    let mut found = HashSet::<String>::new();

    for l in stdout.lines() {
        assert!(l == l.trim());
        if let Some((left, right)) = l.split_once('=') {
            assert!(!left.contains("\""));
            assert!(!right.contains("\""));
        } else {
            assert!(l.contains('='));
        }
        assert!(found.insert(l.to_string()), "{}", &l);
    }

    let Contains { contains, doesnt_contain } = contains;

    {
        let should_found = HashSet::<String>::from_iter(contains.iter().map(|s| s.to_string()));
        let diff: Vec<_> = should_found.difference(&found).collect();
        assert!(diff.is_empty(), "should found: {:?}, didn't found {:?}", &should_found, &diff);
    }
    {
        let should_not_find =
            HashSet::<String>::from_iter(doesnt_contain.iter().map(|s| s.to_string()));
        let diff: Vec<_> = should_not_find.intersection(&found).collect();
        assert!(diff.is_empty(), "should not find {:?}, did found {:?}", &should_not_find, &diff);
    }
}
