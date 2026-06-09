// Check that linking to a framework actually makes it to the linker.

//@ only-apple

use run_make_support::{cmd, rustc};

fn main() {
    rustc().input("dep-link-framework.rs").run();
    rustc().input("dep-link-weak-framework.rs").run();

    rustc().input("empty.rs").run();
    cmd("otool").arg("-L").arg("no-link").run_fail().assert_stdout_not_contains("CoreFoundation");

    rustc().input("link-framework.rs").run();
    cmd("otool")
        .arg("-L")
        .arg("link-framework")
        .run()
        .assert_stdout_contains("CoreFoundation")
        .assert_stdout_not_contains("weak");

    rustc().input("link-weak-framework.rs").run();
    cmd("otool")
        .arg("-L")
        .arg("link-weak-framework")
        .run()
        .assert_stdout_contains("CoreFoundation")
        .assert_stdout_contains("weak");

    // When linking the framework both normally, and weakly, the weak linking takes preference.
    rustc().input("link-both.rs").run();
    cmd("otool")
        .arg("-L")
        .arg("link-both")
        .run()
        .assert_stdout_contains("CoreFoundation")
        .assert_stdout_contains("weak");
}
