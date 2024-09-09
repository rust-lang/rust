//! This checks the output of `--print=check-cfg`

extern crate run_make_support;

use std::collections::HashSet;
use std::iter::FromIterator;

use run_make_support::rustc;

struct CheckCfg {
    args: &'static [&'static str],
    contains: Contains,
}

enum Contains {
    Some { contains: &'static [&'static str], doesnt_contain: &'static [&'static str] },
    Only(&'static str),
}

fn main() {
    check(CheckCfg { args: &[], contains: Contains::Only("any()=any()") });
    check(CheckCfg {
        args: &["--check-cfg=cfg()"],
        contains: Contains::Some {
            contains: &["unix", "miri"],
            doesnt_contain: &["any()", "any()=any()"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(any())"],
        contains: Contains::Some {
            contains: &["any()", "unix", r#"target_feature="crt-static""#],
            doesnt_contain: &["any()=any()"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature)"],
        contains: Contains::Some {
            contains: &["unix", "miri", "feature"],
            doesnt_contain: &["any()", "any()=any()", "feature=none()", "feature="],
        },
    });
    check(CheckCfg {
        args: &[r#"--check-cfg=cfg(feature, values(none(), "", "test", "lol"))"#],
        contains: Contains::Some {
            contains: &["feature", "feature=\"\"", "feature=\"test\"", "feature=\"lol\""],
            doesnt_contain: &["any()", "any()=any()", "feature=none()", "feature="],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature, values())"],
        contains: Contains::Some {
            contains: &["feature="],
            doesnt_contain: &["any()", "any()=any()", "feature=none()", "feature"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature, values())", "--check-cfg=cfg(feature, values(none()))"],
        contains: Contains::Some {
            contains: &["feature"],
            doesnt_contain: &["any()", "any()=any()", "feature=none()", "feature="],
        },
    });
    check(CheckCfg {
        args: &[
            r#"--check-cfg=cfg(feature, values(any()))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        contains: Contains::Some {
            contains: &["unix", "miri", "feature=any()"],
            doesnt_contain: &["any()", "any()=any()", "feature", "feature=", "feature=\"tmp\""],
        },
    });
    check(CheckCfg {
        args: &[
            r#"--check-cfg=cfg(has_foo, has_bar)"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        contains: Contains::Some {
            contains: &["has_foo", "has_bar", "feature=\"tmp\""],
            doesnt_contain: &["any()", "any()=any()", "feature"],
        },
    });
}

fn check(CheckCfg { args, contains }: CheckCfg) {
    let output =
        rustc().input("lib.rs").arg("-Zunstable-options").print("check-cfg").args(args).run();

    let stdout = output.stdout_utf8();

    let mut found = HashSet::<String>::new();

    for l in stdout.lines() {
        assert!(l == l.trim());
        if let Some((left, right)) = l.split_once('=') {
            if right != "any()" && right != "" {
                assert!(right.starts_with("\""));
                assert!(right.ends_with("\""));
            }
            assert!(!left.contains("\""));
        } else {
            assert!(!l.contains("\""));
        }
        assert!(found.insert(l.to_string()), "{}", &l);
    }

    match contains {
        Contains::Some { contains, doesnt_contain } => {
            {
                let should_found =
                    HashSet::<String>::from_iter(contains.iter().map(|s| s.to_string()));
                let diff: Vec<_> = should_found.difference(&found).collect();
                assert!(
                    diff.is_empty(),
                    "should found: {:?}, didn't found {:?}",
                    &should_found,
                    &diff
                );
            }
            {
                let should_not_find =
                    HashSet::<String>::from_iter(doesnt_contain.iter().map(|s| s.to_string()));
                let diff: Vec<_> = should_not_find.intersection(&found).collect();
                assert!(
                    diff.is_empty(),
                    "should not find {:?}, did found {:?}",
                    &should_not_find,
                    &diff
                );
            }
        }
        Contains::Only(only) => {
            assert!(found.contains(&only.to_string()), "{:?} != {:?}", &only, &found);
            assert!(found.len() == 1, "len: {}, instead of 1", found.len());
        }
    }
}
