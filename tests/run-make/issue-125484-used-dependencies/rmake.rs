//@ needs-target-std
//
// Non-regression test for issues #125474, #125484, #125646, with the repro taken from #125484. Some
// queries use "used dependencies" while others use "speculatively loaded dependencies", and an
// indexing ICE appeared in some cases when these were unexpectedly used in the same context.

// FIXME: this should probably be a UI test instead of a run-make test, but I *cannot* find a way to
// make compiletest annotations reproduce the ICE with the minimizations from issues #125474 and
// #125484.

use run_make_support::rustc;

fn main() {
    // The dependency is not itself significant, apart from sharing a name with one of main's
    // modules.
    rustc().crate_name("same").crate_type("rlib").input("dependency.rs").run();

    // Here, an ICE would happen when building the linker command.
    rustc().input("main.rs").extern_("same", "libsame.rlib").run();
}
