//! This checks the output of `--print=check-cfg`

extern crate run_make_support;

use std::collections::HashSet;
use std::iter::FromIterator;
use std::ops::Deref;

use run_make_support::rustc;

fn main() {
    check(
        /*args*/ &[],
        /*has_any*/ false,
        /*has_any_any*/ true,
        /*contains*/ &[],
    );
    check(
        /*args*/ &["--check-cfg=cfg()"],
        /*has_any*/ false,
        /*has_any_any*/ false,
        /*contains*/ &["unix", "miri"],
    );
    check(
        /*args*/ &["--check-cfg=cfg(any())"],
        /*has_any*/ true,
        /*has_any_any*/ false,
        /*contains*/ &["windows", "test"],
    );
    check(
        /*args*/ &["--check-cfg=cfg(feature)"],
        /*has_any*/ false,
        /*has_any_any*/ false,
        /*contains*/ &["unix", "miri", "feature"],
    );
    check(
        /*args*/ &[r#"--check-cfg=cfg(feature, values(none(), "", "test", "lol"))"#],
        /*has_any*/ false,
        /*has_any_any*/ false,
        /*contains*/ &["feature", "feature=\"\"", "feature=\"test\"", "feature=\"lol\""],
    );
    check(
        /*args*/
        &[
            r#"--check-cfg=cfg(feature, values(any()))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        /*has_any*/ false,
        /*has_any_any*/ false,
        /*contains*/ &["unix", "miri", "feature=any()"],
    );
    check(
        /*args*/
        &[
            r#"--check-cfg=cfg(has_foo, has_bar)"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
            r#"--check-cfg=cfg(feature, values("tmp"))"#,
        ],
        /*has_any*/ false,
        /*has_any_any*/ false,
        /*contains*/ &["has_foo", "has_bar", "feature=\"tmp\""],
    );
}

fn check(args: &[&str], has_any: bool, has_any_any: bool, contains: &[&str]) {
    let output = rustc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .arg("--print=check-cfg")
        .args(&*args)
        .run();

    let stdout = String::from_utf8(output.stdout).unwrap();

    let mut found_any = false;
    let mut found_any_any = false;
    let mut found = HashSet::<String>::new();
    let mut recorded = HashSet::<String>::new();

    for l in stdout.lines() {
        assert!(l == l.trim());
        if l == "any()" {
            found_any = true;
        } else if l == "any()=any()" {
            found_any_any = true;
        } else if let Some((left, right)) = l.split_once('=') {
            if right != "any()" && right != "" {
                assert!(right.starts_with("\""));
                assert!(right.ends_with("\""));
            }
            assert!(!left.contains("\""));
        } else {
            assert!(!l.contains("\""));
        }
        assert!(recorded.insert(l.to_string()), "{}", &l);
        if contains.contains(&l) {
            assert!(found.insert(l.to_string()), "{}", &l);
        }
    }

    let should_found = HashSet::<String>::from_iter(contains.iter().map(|s| s.to_string()));
    let diff: Vec<_> = should_found.difference(&found).collect();

    assert_eq!(found_any, has_any);
    assert_eq!(found_any_any, has_any_any);
    assert_eq!(found_any_any, recorded.len() == 1);
    assert!(diff.is_empty(), "{:?} != {:?} (~ {:?})", &should_found, &found, &diff);
}
