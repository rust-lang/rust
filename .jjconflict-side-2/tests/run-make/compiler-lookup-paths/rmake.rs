//@ needs-target-std
//
// Since #19941, rustc can accept specifications on its library search paths.
// This test runs Rust programs with varied library dependencies, expecting them
// to succeed or fail depending on the situation.
// The second part of the tests also checks that libraries with an incorrect hash
// fail to be used by the compiler.
// See https://github.com/rust-lang/rust/pull/19941

//@ ignore-wasm32
//@ ignore-wasm64
// Reason: a C compiler is required for build_native_static_lib

use run_make_support::{build_native_static_lib, rfs, rustc, static_lib_name};

fn main() {
    build_native_static_lib("native");
    let lib_native = static_lib_name("native");
    rfs::create_dir_all("crate");
    rfs::create_dir_all("native");
    rfs::rename(&lib_native, format!("native/{}", &lib_native));
    rustc().input("a.rs").run();
    rfs::rename("liba.rlib", "crate/liba.rlib");
    rustc().input("b.rs").specific_library_search_path("native", "crate").run_fail();
    rustc().input("b.rs").specific_library_search_path("dependency", "crate").run_fail();
    rustc().input("b.rs").specific_library_search_path("crate", "crate").run();
    rustc().input("b.rs").specific_library_search_path("all", "crate").run();

    rustc().input("c.rs").specific_library_search_path("native", "crate").run_fail();
    rustc().input("c.rs").specific_library_search_path("crate", "crate").run_fail();
    rustc().input("c.rs").specific_library_search_path("dependency", "crate").run();
    rustc().input("c.rs").specific_library_search_path("all", "crate").run();

    rustc().input("d.rs").specific_library_search_path("dependency", "native").run_fail();
    rustc().input("d.rs").specific_library_search_path("crate", "native").run_fail();
    rustc().input("d.rs").specific_library_search_path("native", "native").run();
    rustc().input("d.rs").specific_library_search_path("all", "native").run();

    // Deduplication tests.
    rfs::create_dir_all("e1");
    rfs::create_dir_all("e2");

    rustc().input("e.rs").output("e1/libe.rlib").run();
    rustc().input("e.rs").output("e2/libe.rlib").run();
    // If the library hash is correct, compilation should succeed.
    rustc().input("f.rs").library_search_path("e1").library_search_path("e2").run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", "e1")
        .library_search_path("e2")
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", "e1")
        .specific_library_search_path("crate", "e2")
        .run();
    // If the library has a different hash, errors should occur.
    rustc().input("e2.rs").output("e2/libe.rlib").run();
    rustc().input("f.rs").library_search_path("e1").library_search_path("e2").run_fail();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", "e1")
        .library_search_path("e2")
        .run_fail();
    rustc()
        .input("f.rs")
        .specific_library_search_path("crate", "e1")
        .specific_library_search_path("crate", "e2")
        .run_fail();
    // Native and dependency paths do not cause errors.
    rustc()
        .input("f.rs")
        .specific_library_search_path("native", "e1")
        .library_search_path("e2")
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("dependency", "e1")
        .library_search_path("e2")
        .run();
    rustc()
        .input("f.rs")
        .specific_library_search_path("dependency", "e1")
        .specific_library_search_path("crate", "e2")
        .run();
}
