// This test is to check if the warning is emitted when no space
// between `-o` and arg is applied, see issue #142812

//@ ignore-cross-compile
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
            "note: output filename `-o ptimize` is applied instead of a flag named `optimize`",
        );
    rustc()
        .input("main.rs")
        .arg("-o0")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: output filename `-o 0` is applied instead of a flag named `o0`",
        );
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
            "note: output filename `-o ut-dir` is applied instead of a flag named `out-dir`",
        )
        .assert_stderr_contains(
            "help: insert a space between `-o` and `ut-dir` if this is intentional: `-o ut-dir`",
        );
    // test real args by iter CG_OPTIONS
    rustc()
        .input("main.rs")
        .arg("-opt_level")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: output filename `-o pt_level` is applied instead of a flag named `opt_level`",
        )
        .assert_stderr_contains(
            "help: insert a space between `-o` and `pt_level` if this is intentional: `-o pt_level`"
        );
    // separater in-sensitive
    rustc()
        .input("main.rs")
        .arg("-opt-level")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: output filename `-o pt-level` is applied instead of a flag named `opt-level`",
        )
        .assert_stderr_contains(
            "help: insert a space between `-o` and `pt-level` if this is intentional: `-o pt-level`"
        );
    rustc()
        .input("main.rs")
        .arg("-overflow-checks")
        .run()
        .assert_stderr_contains(
            "warning: option `-o` has no space between flag name and value, which can be confusing",
        )
        .assert_stderr_contains(
            "note: output filename `-o verflow-checks` \
            is applied instead of a flag named `overflow-checks`",
        )
        .assert_stderr_contains(
            "help: insert a space between `-o` and `verflow-checks` \
            if this is intentional: `-o verflow-checks`",
        );

    // No warning for Z_OPTIONS
    rustc().input("main.rs").arg("-oom").run().assert_stderr_equals("");

    // test no warning when there is space between `-o` and arg
    rustc().input("main.rs").arg("-o").arg("ptimize").run().assert_stderr_equals("");
    rustc().input("main.rs").arg("--out-dir").arg("xxx").run().assert_stderr_equals("");
    rustc().input("main.rs").arg("-o").arg("out-dir").run().assert_stderr_equals("");
}
