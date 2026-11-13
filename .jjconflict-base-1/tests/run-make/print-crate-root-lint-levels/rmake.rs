//! This checks the output of `--print=crate-root-lint-levels`

extern crate run_make_support;

use std::collections::HashSet;
use std::iter::FromIterator;

use run_make_support::rustc;

struct CrateRootLintLevels {
    args: &'static [&'static str],
    contains: Contains,
}

struct Contains {
    contains: &'static [&'static str],
    doesnt_contain: &'static [&'static str],
}

fn main() {
    check(CrateRootLintLevels {
        args: &[],
        contains: Contains {
            contains: &[
                "unexpected_cfgs=allow",
                "unused_mut=expect",
                "warnings=warn",
                "stable_features=warn",
                "unknown_lints=warn",
            ],
            doesnt_contain: &["unexpected_cfgs=warn", "unused_mut=warn"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Wunexpected_cfgs"],
        contains: Contains {
            contains: &["unexpected_cfgs=allow", "warnings=warn"],
            doesnt_contain: &["unexpected_cfgs=warn"],
        },
    });
    check(CrateRootLintLevels {
        args: &["-Dwarnings"],
        contains: Contains {
            contains: &[
                "unexpected_cfgs=allow",
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
            contains: &["warnings=warn", "stable_features=deny", "unexpected_cfgs=allow"],
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
                "unexpected_cfgs=allow",
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
    let output = rustc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .print("crate-root-lint-levels")
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
