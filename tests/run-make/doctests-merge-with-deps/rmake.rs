//! Since PR <https://github.com/rust-lang/rust/pull/137899>, merged doctests reside in so-called
//! "bundle" crates which are separate from the actual "runner" crates that contain the necessary
//! scaffolding / infrastructure for executing the tests by utilizing the unstable crate `test`
//! and the internal lang feature `rustc_attrs`.
//!
//! In the light of this two-crate setup (per edition), this test ensures that rustdoc can handle
//! mergeable doctests with direct and transitive dependencies (which would become ("first-degree")
//! transitive and "second-degree" transitive dependencies of the runner crates, respectively).
//!
//! While this is about doctest merging which is only available in Rust 2024 and beyond,
//! we also test Rust 2021 here to ensure that in this specific scenario rustdoc doesn't
//! grossly diverge in observable behavior.

use run_make_support::{bare_rustc, rustdoc};

fn main() {
    // Re. `bare_rustc` over `rustc` (which would implicitly add `-L.`):
    // This test is all about verifying that rustdoc is able to find dependencies
    // and properly propagates library search paths etc.
    // Anything implicit like that would only obfuscate this test or even
    // accidentally break it.

    //
    // Build crate `a_a` and its dependent `a` which is the direct dependency of
    // the doctests inside `doctest.rs` *and* of crate `doctest` itself!
    //

    bare_rustc().current_dir("deps").input("dep_a_a.rs").crate_type("lib").run();

    bare_rustc().input("dep_a.rs").crate_type("lib").args(["--extern=dep_a_a", "-Ldeps"]).run();

    //
    // Build crate `b_b` and its dependent `b` which is the direct dependency of
    // the first doctest in `doctest.rs` *but not* of crate `doctest` itself!
    //

    bare_rustc().current_dir("deps").input("dep_b_b.rs").crate_type("lib").run();

    bare_rustc().input("dep_b.rs").crate_type("lib").args(["--extern=dep_b_b", "-Ldeps"]).run();

    //
    // Collect and run the doctests inside `doctest.rs`.
    //

    for edition in ["2021", "2024"] {
        // NB: `-Ldependency=<path>` only adds *transitive* dependencies to
        //     the search path contrary to `-L<path>`.

        rustdoc()
            .input("doctest.rs")
            .crate_type("lib")
            .edition(edition)
            // Adds crate `a` as a dep of `doctest` *and* its contained doctests.
            // Also registers transitive dependencies.
            .extern_("dep_a", "libdep_a.rlib")
            .arg("-Ldependency=deps")
            // Adds crate `b` as a dep of `doctest`'s contained doctests.
            // Also registers transitive dependencies.
            .args([
                "-Zunstable-options",
                "--doctest-compilation-args",
                "--extern=dep_b=libdep_b.rlib -Ldependency=deps",
            ])
            .arg("--test")
            .arg("--test-args=--test-threads=1")
            .run();
    }
}
