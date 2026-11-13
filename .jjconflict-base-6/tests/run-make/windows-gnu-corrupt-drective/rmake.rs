//@ only-windows-gnu

use run_make_support::{bare_rustc, rustc};

fn main() {
    // bare_rustc so that this doesn't try to cross-compile our linker
    bare_rustc().input("fake-linker.rs").output("fake-linker").run();
    rustc()
        .input("main.rs")
        .linker("./fake-linker")
        .arg("-Wlinker-messages")
        .run()
        .assert_stderr_contains("Warning: .drectve");
}
