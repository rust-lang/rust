// Check that linking to a framework actually makes it to the linker.

//@ only-apple

use run_make_support::{cmd, rustc};

fn main() {
    rustc().input("dep-link-framework.rs").run();
    rustc().input("dep-link-weak-framework.rs").run();

    rustc().input("empty.rs").run();
    cmd("otool").arg("-L").arg("no-link").run().assert_stdout_not_contains("CoreFoundation");

    rustc().input("link-framework.rs").run();
    let out = cmd("otool").arg("-L").arg("link-framework").run();
    out.assert_stdout_contains("CoreFoundation");
    out.assert_stdout_not_contains("weak");

    rustc().input("link-weak-framework.rs").run();
    let out = cmd("otool").arg("-L").arg("link-weak-framework").run();
    out.assert_stdout_contains("CoreFoundation");
    out.assert_stdout_contains("weak");

    // When linking the framework both normally, and weakly, the weak linking takes preference.
    rustc().input("link-both.rs").run();
    let out = cmd("otool").arg("-L").arg("link-both").run();
    out.assert_stdout_contains("CoreFoundation");
    out.assert_stdout_contains("weak");
}
