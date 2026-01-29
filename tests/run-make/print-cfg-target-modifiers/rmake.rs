//! This checks the output of `--print=cfg` to confirm that cfgs are correctly set for
//! target modifier flags

// ignore-tidy-linelength
//@ needs-llvm-components: aarch64 x86
// Note: without the needs-llvm-components it will fail on LLVM built without the required
// components listed above.

use std::collections::HashSet;
use std::iter::FromIterator;

use run_make_support::rustc;

struct PrintCfg {
    target: &'static str,
    flag: &'static str,
    cfgs: &'static [&'static str],
}

fn main() {
    // `-Zfixed-x18`
    check(PrintCfg {
        target: "aarch64-unknown-linux-gnu",
        flag: "-Zfixed-x18",
        cfgs: &["target_modifier_fixed_x18"],
    });
    // `-Zindirect-branch-cs-prefix`
    check(PrintCfg {
        target: "x86_64-unknown-linux-gnu",
        flag: "-Zindirect-branch-cs-prefix",
        cfgs: &["target_modifier_indirect_branch_cs_prefix"],
    });
    // `-Zreg-struct-return`
    check(PrintCfg {
        target: "i686-unknown-linux-gnu",
        flag: "-Zreg-struct-return",
        cfgs: &["target_modifier_reg_struct_return"],
    });
    // `-Zregparm`
    check(PrintCfg {
        target: "i686-unknown-linux-gnu",
        flag: "-Zregparm=0",
        cfgs: &["target_modifier_regparm=\"0\""],
    });
    // `-Zretpoline`/`-Zretpoline-external-thunk`
    check(PrintCfg {
        target: "x86_64-unknown-linux-gnu",
        flag: "-Zretpoline",
        cfgs: &["target_modifier_retpoline"],
    });
    check(PrintCfg {
        target: "x86_64-unknown-linux-gnu",
        flag: "-Zretpoline-external-thunk",
        cfgs: &["target_modifier_retpoline_external_thunk"],
    });
}

fn check(PrintCfg { target, flag, cfgs }: PrintCfg) {
    let output = rustc().target(target).arg(flag).print("cfg").run();
    let stdout = output.stdout_utf8();

    let mut found = HashSet::<String>::new();
    let mut recorded = HashSet::<String>::new();

    for l in stdout.lines() {
        assert!(l == l.trim());
        if let Some((left, right)) = l.split_once('=') {
            assert!(right.starts_with("\""));
            assert!(right.ends_with("\""));
            assert!(!left.contains("\""));
        } else {
            assert!(!l.contains("\""));
        }

        assert!(recorded.insert(l.to_string()), "duplicated: {}", &l);
        if cfgs.contains(&l) {
            assert!(found.insert(l.to_string()), "duplicated (includes): {}", &l);
        }
    }

    let should_found = HashSet::<String>::from_iter(cfgs.iter().map(|s| s.to_string()));
    let diff: Vec<_> = should_found.difference(&found).collect();

    assert!(diff.is_empty(), "expected: {:?}, found: {:?} (~ {:?})", &should_found, &found, &diff);
}
