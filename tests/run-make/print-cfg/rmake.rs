//! This checks the output of `--print=cfg`
//!
//! Specifically it checks that output is correctly formatted
//! (ie. no duplicated cfgs, values are between "", names are not).
//!
//! It also checks that some targets have the correct set cfgs.

extern crate run_make_support;

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::BufRead;
use std::iter::FromIterator;

use run_make_support::{rustc, tmp_dir};

fn main() {
    check(
        /*target*/ "x86_64-pc-windows-gnu",
        /*includes*/ &["windows", "target_arch=\"x86_64\""],
        /*disallow*/ &["unix"],
    );
    check(
        /*target*/ "i686-pc-windows-msvc",
        /*includes*/ &["windows", "target_env=\"msvc\""],
        /*disallow*/ &["unix"],
    );
    check(
        /*target*/ "i686-apple-darwin",
        /*includes*/ &["unix", "target_os=\"macos\"", "target_vendor=\"apple\""],
        /*disallow*/ &["windows"],
    );
    check(
        /*target*/ "i686-unknown-linux-gnu",
        /*includes*/ &["unix", "target_env=\"gnu\""],
        /*disallow*/ &["windows"],
    );
    check(
        /*target*/ "arm-unknown-linux-gnueabihf",
        /*includes*/ &["unix", "target_abi=\"eabihf\""],
        /*disallow*/ &["windows"],
    );
}

fn check(target: &str, includes: &[&str], disallow: &[&str]) {
    fn _inner(output: &str, includes: &[&str], disallow: &[&str]) {
        let mut found = HashSet::<String>::new();
        let mut recorded = HashSet::<String>::new();

        for l in output.lines() {
            assert!(l == l.trim());
            if let Some((left, right)) = l.split_once('=') {
                assert!(right.starts_with("\""));
                assert!(right.ends_with("\""));
                assert!(!left.contains("\""));
            } else {
                assert!(!l.contains("\""));
            }

            assert!(recorded.insert(l.to_string()), "duplicated: {}", &l);
            assert!(!disallow.contains(&l), "found disallowed: {}", &l);
            if includes.contains(&l) {
                assert!(found.insert(l.to_string()), "duplicated (includes): {}", &l);
            }
        }

        let should_found = HashSet::<String>::from_iter(includes.iter().map(|s| s.to_string()));
        let diff: Vec<_> = should_found.difference(&found).collect();

        assert!(
            diff.is_empty(),
            "expected: {:?}, found: {:?} (~ {:?})",
            &should_found,
            &found,
            &diff
        );
    }

    // --print=cfg
    {
        let output = rustc().target(target).print("cfg").run();

        let stdout = String::from_utf8(output.stdout).unwrap();

        _inner(&stdout, includes, disallow);
    }

    // --print=cfg=PATH
    {
        let tmp_path = tmp_dir().join(format!("{target}.cfg"));
        let mut print_arg = OsString::from("--print=cfg=");
        print_arg.push(tmp_path.as_os_str());

        let output = rustc().target(target).arg(print_arg).run();

        let output = std::fs::read_to_string(&tmp_path).unwrap();

        _inner(&output, includes, disallow);
    }
}
