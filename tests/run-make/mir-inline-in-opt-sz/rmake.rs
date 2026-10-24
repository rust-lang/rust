// The MIR inliner must run at `-Copt-level=s`/`z` exactly as it does at `-Copt-level=2`: all three
// derive a MIR opt level of 2. `-Zdump-mir=Inline` dumps only when the pass runs, so its absence is
// the probe.
//
// The `-Copt-level=0`/`1` and explicit `-Z` cases are the control group: without them the test
// would still pass if `-Zdump-mir=Inline` started dumping unconditionally. The `attr` cases check
// that `#[optimize]` on an item may only lower what is done to that body, never raise it.

use std::path::Path;

use run_make_support::{rfs, rustc};

struct Case {
    name: &'static str,
    input: &'static str,
    flags: &'static [&'static str],
    expect_inline_dump: bool,
}

fn main() {
    let cases = [
        // At MIR opt level 2 the inliner runs, whatever the objective.
        Case {
            name: "speed2",
            input: "main.rs",
            flags: &["-Copt-level=2"],
            expect_inline_dump: true,
        },
        Case {
            name: "speed3",
            input: "main.rs",
            flags: &["-Copt-level=3"],
            expect_inline_dump: true,
        },
        Case {
            name: "size_s",
            input: "main.rs",
            flags: &["-Copt-level=s"],
            expect_inline_dump: true,
        },
        Case {
            name: "size_z",
            input: "main.rs",
            flags: &["-Copt-level=z"],
            expect_inline_dump: true,
        },
        // Below MIR opt level 2 there is nothing to inline at, so the pass must stay off...
        Case {
            name: "none0",
            input: "main.rs",
            flags: &["-Copt-level=0"],
            expect_inline_dump: false,
        },
        Case {
            name: "less1",
            input: "main.rs",
            flags: &["-Copt-level=1"],
            expect_inline_dump: false,
        },
        // ...and an explicit `-Z` must keep winning over `-Copt-level`, in both directions.
        Case {
            name: "size_z_mir_opt_level0",
            input: "main.rs",
            flags: &["-Copt-level=z", "-Zmir-opt-level=0"],
            expect_inline_dump: false,
        },
        Case {
            name: "size_z_mir_opt_level1",
            input: "main.rs",
            flags: &["-Copt-level=z", "-Zmir-opt-level=1"],
            expect_inline_dump: false,
        },
        Case {
            name: "size_z_inline_mir_no",
            input: "main.rs",
            flags: &["-Copt-level=z", "-Zinline-mir=no"],
            expect_inline_dump: false,
        },
        Case {
            name: "none0_inline_mir_yes",
            input: "main.rs",
            flags: &["-Copt-level=0", "-Zinline-mir=yes"],
            expect_inline_dump: true,
        },
        // `#[optimize]` on an item does not decide the compilation's level.
        Case {
            name: "attr_size_none0",
            input: "attr.rs",
            flags: &["-Copt-level=0"],
            expect_inline_dump: false,
        },
        Case {
            name: "attr_speed_none0",
            input: "attr.rs",
            flags: &["-Copt-level=0", "-Zmir-opt-level=0"],
            expect_inline_dump: false,
        },
        Case {
            name: "attr_size_less1",
            input: "attr.rs",
            flags: &["-Copt-level=1"],
            expect_inline_dump: false,
        },
        Case {
            name: "attr_size_size_z",
            input: "attr.rs",
            flags: &["-Copt-level=z"],
            expect_inline_dump: true,
        },
    ];

    for case in cases {
        let dumped = compile_case(case.name, case.input, case.flags);
        assert_eq!(
            dumped, case.expect_inline_dump,
            "{}: unexpected Inline dump presence",
            case.name
        );
    }
}

fn compile_case(dump_dir: &str, input: &str, flags: &[&str]) -> bool {
    if Path::new(dump_dir).exists() {
        rfs::remove_dir_all(dump_dir);
    }
    rfs::create_dir_all(dump_dir);

    let mut cmd = rustc();
    cmd.input(input)
        .arg("--emit=mir")
        .arg("-Zdump-mir=Inline")
        .arg(format!("-Zdump-mir-dir={dump_dir}"));
    for &flag in flags {
        cmd.arg(flag);
    }
    cmd.run();

    has_inline_dump_file(dump_dir)
}

fn has_inline_dump_file(dir: &str) -> bool {
    rfs::read_dir(dir)
        .flatten()
        .any(|entry| entry.file_name().to_string_lossy().contains(".Inline."))
}
