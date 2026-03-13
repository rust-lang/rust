use std::path::Path;

use run_make_support::{CompletedProcess, rfs, rustc};

struct Case {
    name: &'static str,
    flags: &'static [&'static str],
    expect_inline_dump: bool,
    expect_running: ExpectedCount,
    expect_not_running: ExpectedCount,
}

enum ExpectedCount {
    Exactly(usize),
    AtLeastOne,
    Zero,
}

fn main() {
    let cases = [
        Case {
            name: "limit0",
            flags: &["-Zmir-opt-bisect-limit=0"],
            expect_inline_dump: false,
            expect_running: ExpectedCount::Exactly(0),
            expect_not_running: ExpectedCount::AtLeastOne,
        },
        Case {
            name: "limit1",
            flags: &["-Zmir-opt-bisect-limit=1"],
            expect_inline_dump: false,
            expect_running: ExpectedCount::Exactly(1),
            expect_not_running: ExpectedCount::AtLeastOne,
        },
        Case {
            name: "huge_limit",
            flags: &["-Zmir-opt-bisect-limit=1000000000"],
            expect_inline_dump: true,
            expect_running: ExpectedCount::AtLeastOne,
            expect_not_running: ExpectedCount::Zero,
        },
        Case {
            name: "limit0_with_force_enable_inline",
            flags: &["-Zmir-opt-bisect-limit=0", "-Zmir-enable-passes=+Inline"],
            expect_inline_dump: false,
            expect_running: ExpectedCount::Exactly(0),
            expect_not_running: ExpectedCount::AtLeastOne,
        },
    ];

    for case in cases {
        let (inline_dumped, running_count, not_running_count, output) =
            compile_case(case.name, case.flags);

        assert_eq!(
            inline_dumped, case.expect_inline_dump,
            "{}: unexpected Inline dump presence",
            case.name
        );

        assert_expected_count(
            running_count,
            case.expect_running,
            &format!("{}: running count", case.name),
        );
        assert_expected_count(
            not_running_count,
            case.expect_not_running,
            &format!("{}: NOT running count", case.name),
        );
    }
}

fn compile_case(dump_dir: &str, extra_flags: &[&str]) -> (bool, usize, usize, CompletedProcess) {
    if Path::new(dump_dir).exists() {
        rfs::remove_dir_all(dump_dir);
    }
    rfs::create_dir_all(dump_dir);

    let mut cmd = rustc();
    cmd.input("main.rs")
        .arg("--emit=mir")
        .arg("-Zmir-opt-level=2")
        .arg("-Copt-level=2")
        .arg("-Zthreads=1")
        .arg("-Zdump-mir=Inline")
        .arg(format!("-Zdump-mir-dir={dump_dir}"));

    for &flag in extra_flags {
        cmd.arg(flag);
    }

    let output = cmd.run();
    let (running_count, not_running_count) = bisect_line_counts(&output);
    (has_inline_dump_file(dump_dir), running_count, not_running_count, output)
}

fn assert_expected_count(actual: usize, expected: ExpectedCount, context: &str) {
    match expected {
        ExpectedCount::Exactly(n) => assert_eq!(actual, n, "{context}"),
        ExpectedCount::AtLeastOne => assert!(actual > 0, "{context}"),
        ExpectedCount::Zero => assert_eq!(actual, 0, "{context}"),
    }
}

fn has_inline_dump_file(dir: &str) -> bool {
    rfs::read_dir(dir)
        .flatten()
        .any(|entry| entry.file_name().to_string_lossy().contains(".Inline."))
}

fn bisect_line_counts(output: &CompletedProcess) -> (usize, usize) {
    let stderr = output.stderr_utf8();
    let running_count =
        stderr.lines().filter(|line| line.starts_with("BISECT: running pass (")).count();
    let not_running_count =
        stderr.lines().filter(|line| line.starts_with("BISECT: NOT running pass (")).count();
    (running_count, not_running_count)
}
