//! This checks the output of `--print=cfg`
//!
//! Specifically it checks that output is correctly formatted
//! (ie. no duplicated cfgs, values are between "", names are not).
//!
//! It also checks that some targets have the correct set cfgs.

// ignore-tidy-linelength
//@ needs-llvm-components: arm x86
// Note: without the needs-llvm-components it will fail on LLVM built without the required
// components listed above.

use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::PathBuf;

use run_make_support::{rfs, rustc};

struct PrintCfg {
    target: &'static str,
    includes: &'static [&'static str],
    disallow: &'static [&'static str],
}

fn main() {
    check(PrintCfg {
        target: "x86_64-pc-windows-gnu",
        includes: &["windows", "target_arch=\"x86_64\""],
        disallow: &["unix"],
    });
    check(PrintCfg {
        target: "i686-pc-windows-msvc",
        includes: &["windows", "target_env=\"msvc\""],
        disallow: &["unix"],
    });
    check(PrintCfg {
        target: "i686-apple-darwin",
        includes: &["unix", "target_os=\"macos\"", "target_vendor=\"apple\""],
        disallow: &["windows"],
    });
    check(PrintCfg {
        target: "i686-unknown-linux-gnu",
        includes: &["unix", "target_env=\"gnu\""],
        disallow: &["windows"],
    });
    check(PrintCfg {
        target: "arm-unknown-linux-gnueabihf",
        includes: &["unix", "target_abi=\"eabihf\""],
        disallow: &["windows"],
    });
}

fn check(PrintCfg { target, includes, disallow }: PrintCfg) {
    fn check_(output: &str, includes: &[&str], disallow: &[&str]) {
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
        let stdout = output.stdout_utf8();

        check_(&stdout, includes, disallow);
    }

    // --print=cfg=PATH
    {
        let tmp_path = PathBuf::from(format!("{target}.cfg"));

        rustc().target(target).print(&format!("cfg={}", tmp_path.display())).run();

        let output = rfs::read_to_string(&tmp_path);

        check_(&output, includes, disallow);
    }
}
