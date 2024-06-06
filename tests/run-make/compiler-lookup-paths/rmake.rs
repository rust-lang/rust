// Since #19941, rustc can accept specifications on its library search paths.
// This test runs Rust programs with varied library dependencies, expecting them
// to succeed or fail depending on the situation.
// The second part of the tests also checks that libraries with an incorrect hash
// fail to be used by the compiler.
// See https://github.com/rust-lang/rust/pull/19941

use run_make_support::fs_wrapper;
use run_make_support::{rmake_out_path, rustc};

fn main() {
    assert!(rmake_out_path("libnative.a").exists());
    fs_wrapper::create_dir_all(rmake_out_path("crate"));
    fs_wrapper::create_dir_all(rmake_out_path("native"));
    fs_wrapper::rename(rmake_out_path("libnative.a"), rmake_out_path("native"));
    rustc().input("a.rs").run();
    fs_wrapper::rename(rmake_out_path("liba.a"), rmake_out_path("crate"));
    rustc()
        .input("b.rs")
        .specific_library_search_path("native", rmake_out_path("crate"))
        .run_fail();
    rustc()
        .input("b.rs")
        .specific_library_search_path("dependency", rmake_out_path("crate"))
        .run_fail();
    rustc().input("b.rs").specific_library_search_path("crate", rmake_out_path("crate")).run();
    rustc().input("b.rs").specific_library_search_path("all", rmake_out_path("crate")).run();

    rustc()
        .input("c.rs")
        .specific_library_search_path("native", rmake_out_path("crate"))
        .run_fail();
    rustc().input("c.rs").specific_library_search_path("crate", rmake_out_path("crate")).run_fail();
    rustc().input("c.rs").specific_library_search_path("dependency", rmake_out_path("crate")).run();
    rustc().input("c.rs").specific_library_search_path("all", rmake_out_path("crate")).run();

    rustc()
        .input("d.rs")
        .specific_library_search_path("dependency", rmake_out_path("native"))
        .run_fail();
    rustc()
        .input("d.rs")
        .specific_library_search_path("crate", rmake_out_path("native"))
        .run_fail();
    rustc().input("d.rs").specific_library_search_path("native", rmake_out_path("native")).run();
    rustc().input("d.rs").specific_library_search_path("all", rmake_out_path("native")).run();

    // Deduplication tests.
    fs_wrapper::create_dir_all(rmake_out_path("e1"));
    fs_wrapper::create_dir_all(rmake_out_path("e2"));

    rustc().input("e.rs").output(rmake_out_path("e1/libe.rlib")).run();
    rustc().input("e.rs").output(rmake_out_path("e2/libe.rlib")).run();
    // If the library hash is correct, compilation should succeed.
    rustc()
        .input("f.rs")
        .library_search_path(rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", rmake_out_path("e1"))
        .specific_library_search_path("crate", rmake_out_path("e2"))
        .run();
    // If the library has a different hash, errors should occur.
    rustc().input("e2.rs").output(rmake_out_path("e2/libe.rlib")).run();
    rustc()
        .input("f.rs")
        .library_search_path(rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run_fail();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run_fail();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", rmake_out_path("e1"))
        .specific_library_search_path("crate", rmake_out_path("e2"))
        .run_fail();
    // Native and dependency paths do not cause errors.
    rustc()
        .input("f.rs")
        .specific_library_search_path("native", rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("dependency", rmake_out_path("e1"))
        .library_search_path(rmake_out_path("e2"))
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("dependency", rmake_out_path("e1"))
        .specific_library_search_path("crate", rmake_out_path("e2"))
        .run();
}
