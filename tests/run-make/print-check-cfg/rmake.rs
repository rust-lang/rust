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
    Nothing,
}

fn main() {
    check(CheckCfg { args: &[], contains: Contains::Nothing });
    check(CheckCfg {
        args: &["--check-cfg=cfg()"],
        contains: Contains::Some {
            contains: &["cfg(unix, values(none()))", "cfg(miri, values(none()))"],
            doesnt_contain: &["cfg(any())"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(any())"],
        contains: Contains::Some {
            contains: &["cfg(any())", "cfg(unix, values(none()))"],
            doesnt_contain: &["any()=any()"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature)"],
        contains: Contains::Some {
            contains: &[
                "cfg(unix, values(none()))",
                "cfg(miri, values(none()))",
                "cfg(feature, values(none()))",
            ],
            doesnt_contain: &["cfg(any())", "cfg(feature)"],
        },
    });
    check(CheckCfg {
        args: &[r#"--check-cfg=cfg(feature, values(none(), "", "test", "lol"))"#],
        contains: Contains::Some {
            contains: &[r#"cfg(feature, values("", "lol", "test", none()))"#],
            doesnt_contain: &["cfg(any())", "cfg(feature, values(none()))", "cfg(feature)"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature, values())"],
        contains: Contains::Some {
            contains: &["cfg(feature, values())"],
            doesnt_contain: &["cfg(any())", "cfg(feature, values(none()))", "cfg(feature)"],
        },
    });
    check(CheckCfg {
        args: &["--check-cfg=cfg(feature, values())", "--check-cfg=cfg(feature, values(none()))"],
        contains: Contains::Some {
            contains: &["cfg(feature, values(none()))"],
            doesnt_contain: &["cfg(any())", "cfg(feature, values())"],
        },
    });
    check(CheckCfg {
        args: &[
            r#"--check-cfg=cfg(feature, values(any()))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        contains: Contains::Some {
            contains: &["cfg(feature, values(any()))"],
            doesnt_contain: &["cfg(any())", r#"cfg(feature, values("tmp"))"#],
        },
    });
    check(CheckCfg {
        args: &[
            r#"--check-cfg=cfg(has_foo, has_bar)"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        contains: Contains::Some {
            contains: &[
                "cfg(has_foo, values(none()))",
                "cfg(has_bar, values(none()))",
                r#"cfg(feature, values("tmp"))"#,
            ],
            doesnt_contain: &["cfg(any())", "cfg(feature)"],
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
        assert!(l.starts_with("cfg("), "{l}");
        assert!(l.ends_with(")"), "{l}");
        assert_eq!(
            l.chars().filter(|c| *c == '(').count(),
            l.chars().filter(|c| *c == ')').count(),
            "{l}"
        );
        assert!(l.chars().filter(|c| *c == '"').count() % 2 == 0, "{l}");
        assert!(found.insert(l.to_string()), "{l}");
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
        Contains::Nothing => {
            assert!(found.len() == 0, "len: {}, instead of 0", found.len());
        }
    }
}
