//@ needs-target-std
//
// Non-regression test for issue #132920 where multiple versions of the same crate are present in
// the dependency graph, and an unexpected error in a dependent crate caused an ICE in the
// unsatisfied bounds diagnostics for traits present in multiple crate versions.
//
// Setup:
// - two versions of the same crate: minibevy_a and minibevy_b
// - minirapier: depends on minibevy_a
// - repro: depends on minirapier and minibevy_b

use run_make_support::rustc;

fn main() {
    // Prepare dependencies, mimicking a check build with cargo.
    rustc()
        .input("minibevy.rs")
        .crate_name("minibevy")
        .crate_type("lib")
        .emit("metadata")
        .metadata("a")
        .extra_filename("-a")
        .run();
    rustc()
        .input("minibevy.rs")
        .crate_name("minibevy")
        .crate_type("lib")
        .emit("metadata")
        .metadata("b")
        .extra_filename("-b")
        .run();
    rustc()
        .input("minirapier.rs")
        .crate_name("minirapier")
        .crate_type("lib")
        .emit("metadata")
        .extern_("minibevy", "libminibevy-a.rmeta")
        .run();

    // Building the main crate used to ICE here when printing the `type annotations needed` error.
    rustc()
        .input("repro.rs")
        .extern_("minibevy", "libminibevy-b.rmeta")
        .extern_("minirapier", "libminirapier.rmeta")
        .run_fail()
        .assert_stderr_not_contains("error: the compiler unexpectedly panicked. this is a bug");
}
