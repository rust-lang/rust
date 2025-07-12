// This test is to check if the warning is emitted when no space
// between `-o` and arg is applied, see issue #142812
use run_make_support::rustc;

fn main() {
    // test fake args
    rustc()
        .input("main.rs")
        .arg("-optimize")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: option `-o ptimize` is applied instead of a flag named `optimize`",
        )
        .assert_stderr_contains("to specify output filename `ptimize`");
    rustc()
        .input("main.rs")
        .arg("-o0")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains("note: option `-o 0` is applied instead of a flag named `o0`")
        .assert_stderr_contains("to specify output filename `0`");
    rustc().input("main.rs").arg("-o1").run();
    // test real args by iter optgroups
    rustc()
        .input("main.rs")
        .arg("-out-dir")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: option `-o ut-dir` is applied instead of a flag named `out-dir`",
        )
        .assert_stderr_contains("to specify output filename `ut-dir`")
        .assert_stderr_contains("Do you mean `--out-dir`?");
    // test real args by iter CG_OPTIONS
    rustc()
        .input("main.rs")
        .arg("-opt-level")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: option `-o pt-level` is applied instead of a flag named `opt-level`",
        )
        .assert_stderr_contains("to specify output filename `pt-level`")
        .assert_stderr_contains("Do you mean `-C opt_level`?");
    rustc()
        .input("main.rs")
        .arg("-overflow-checks")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: option `-o verflow-checks` is applied instead of a flag named `overflow-checks`",
        )
        .assert_stderr_contains("to specify output filename `verflow-checks`")
        .assert_stderr_contains("Do you mean `-C overflow_checks`?");
    // test real args by iter Z_OPTIONS
    rustc()
        .input("main.rs")
        .arg("-oom")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains("note: option `-o om` is applied instead of a flag named `oom`")
        .assert_stderr_contains("to specify output filename `om`")
        .assert_stderr_contains("note: Do you mean `-Z oom`?");

    // test no warning when there is space between `-o` and arg
    rustc().input("main.rs").arg("-o").arg("ptimize").run().assert_stderr_equals("");
    rustc().input("main.rs").arg("--out-dir").arg("xxx").run().assert_stderr_equals("");
    rustc().input("main.rs").arg("-o").arg("out-dir").run().assert_stderr_equals("");
}
