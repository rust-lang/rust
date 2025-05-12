//@ ignore-cross-compile

use run_make_support::rustc;

// FIXME: it would be good to check that it's actually the rightmost flags
// that are used when multiple flags are specified, but I can't think of a
// reliable way to check this.
fn main() {
    // Test that `-O` and `-C opt-level` can be specified multiple times.
    // The rightmost flag will be used over any previous flags.
    rustc().arg("-O").arg("-O").input("main.rs").run();
    rustc().arg("-O").arg("-C").arg("opt-level=0").input("main.rs").run();
    rustc().arg("-C").arg("opt-level=0").arg("-O").input("main.rs").run();
    rustc().arg("-C").arg("opt-level=0").arg("-C").arg("opt-level=2").input("main.rs").run();
    rustc().arg("-C").arg("opt-level=2").arg("-C").arg("opt-level=0").input("main.rs").run();

    // Test that `-g` and `-C debuginfo` can be specified multiple times.
    // The rightmost flag will be used over any previous flags.
    rustc().arg("-g").arg("-g").input("main.rs").run();
    rustc().arg("-g").arg("-C").arg("debuginfo=0").input("main.rs").run();
    rustc().arg("-C").arg("debuginfo=0").arg("-g").input("main.rs").run();
    rustc().arg("-C").arg("debuginfo=0").arg("-C").arg("debuginfo=2").input("main.rs").run();
    rustc().arg("-C").arg("debuginfo=2").arg("-C").arg("debuginfo=0").input("main.rs").run();
}
