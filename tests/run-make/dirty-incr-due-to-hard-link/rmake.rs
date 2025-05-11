//@ only-x86_64-unknown-linux-gnu

// Regression test for the incremental bug in <https://github.com/rust-lang/rust/issues/139407>.
//
// A detailed explanation is described in <https://github.com/rust-lang/rust/pull/139453>,
// however the gist of the issue is that hard-linking temporary files can interact strangely
// across incremental sessions that are not finalized due to errors originating from the
// codegen backend.

use run_make_support::{run, rustc};

fn main() {
    let mk_rustc = || {
        let mut rustc = rustc();
        rustc.input("test.rs").incremental("incr").arg("-Csave-temps").output("test");
        rustc
    };

    // Revision 1
    mk_rustc().cfg("rpass1").run();

    run("test");

    // Revision 2
    mk_rustc().cfg("cfail2").run_fail();
    // Expected to fail.

    // Revision 3
    mk_rustc().cfg("rpass3").run();

    run("test");
}
