// This test case makes sure that two identical invocations of the compiler
// (i.e. same code base, same compile-flags, same compiler-versions, etc.)
// produce the same output. In the past, symbol names of monomorphized functions
// were not deterministic (which we want to avoid).
//
// The test tries to exercise as many different paths into symbol name
// generation as possible:
//
// - regular functions
// - generic functions
// - methods
// - statics
// - closures
// - enum variant constructors
// - tuple struct constructors
// - drop glue
// - FnOnce adapters
// - Trait object shims
// - Fn Pointer shims
// See https://github.com/rust-lang/rust/pull/32293

// FIXME(Oneirical): ignore-musl
// FIXME(Oneirical): two of these test blocks will apparently fail on windows
// FIXME(Oneirical): try it on test-various
// # FIXME: Builds of `bin` crate types are not deterministic with debuginfo=2 on
// # Windows.
// # See: https://github.com/rust-lang/rust/pull/87320#issuecomment-920105533
// # Issue: https://github.com/rust-lang/rust/issues/88982

use run_make_support::{bin_name, cwd, diff, rfs, run_in_tmpdir, rust_lib_name, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc().input("linker.rs").opt().run();
        rustc().input("reproducible-build-aux.rs").run();
        rustc()
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        rustc()
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        diff().actual_file("linker-arguments1").expected_file("linker-arguments2").run();
    });

    run_in_tmpdir(|| {
        rustc().input("linker.rs").opt().run();
        rustc().arg("-g").input("reproducible-build-aux.rs").run();
        rustc()
            .arg("-g")
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        rustc()
            .arg("-g")
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        diff().actual_file("linker-arguments1").expected_file("linker-arguments2").run();
    });

    run_in_tmpdir(|| {
        rustc().input("linker.rs").opt().run();
        rustc().opt().input("reproducible-build-aux.rs").run();
        rustc()
            .opt()
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        rustc()
            .opt()
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        diff().actual_file("linker-arguments1").expected_file("linker-arguments2").run();
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rustc().input("reproducible-build.rs").crate_type("rlib").library_search_path("b").run();
        rfs::copy(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
        rustc().input("reproducible-build.rs").crate_type("rlib").library_search_path("a").run();
        assert_eq!(rfs::read(rust_lib_name("reproducible_build")), rfs::read(rust_lib_name("foo")));
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .arg("--remap-path-prefix=/a=/c")
            .run();
        rfs::copy(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .arg("--remap-path-prefix=/b=/c")
            .run();
        assert_eq!(rfs::read(rust_lib_name("reproducible_build")), rfs::read(rust_lib_name("foo")));
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        rustc()
            .input("reproducible-build.rs")
            .crate_type("bin")
            .arg(&format!("--remap-path-prefix={}=/b", cwd().display()))
            .run();
        eprintln!("{:#?}", rfs::shallow_find_dir_entries(cwd()));
        rfs::copy(bin_name("reproducible_build"), bin_name("foo"));
        rustc()
            .input("test/reproducible-build.rs")
            .crate_type("bin")
            .arg("--remap-path-prefix=/test=/b")
            .run();
        assert_eq!(rfs::read(bin_name("reproducible_build")), rfs::read(bin_name("foo")));
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .arg(&format!("--remap-path-prefix={}=/b", cwd().display()))
            .run();
        rfs::copy(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
        rustc()
            .input("test/reproducible-build.rs")
            .crate_type("rlib")
            .arg("--remap-path-prefix=/test=/b")
            .run();
        assert_eq!(rfs::read(rust_lib_name("reproducible_build")), rfs::read(rust_lib_name("foo")));
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        rustc()
            .input("reproducible-build.rs")
            .crate_type("bin")
            .arg("-Zremap-path-prefix=.")
            .arg("-Cdebuginfo=2")
            .run();
        rfs::copy(bin_name("reproducible_build"), bin_name("first"));
        rustc()
            .input("test/reproducible-build.rs")
            .crate_type("bin")
            .arg("-Zremap-path-prefix=.")
            .arg("-Cdebuginfo=2")
            .run();
        assert_eq!(rfs::read(bin_name("first")), rfs::read(bin_name("reproducible_build")));
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .arg("-Zremap-path-prefix=.")
            .arg("-Cdebuginfo=2")
            .run();
        rfs::copy("reproducible_build", "first");
        rustc()
            .input("test/reproducible-build.rs")
            .crate_type("rlib")
            .arg("-Zremap-path-prefix=.")
            .arg("-Cdebuginfo=2")
            .run();
        assert_eq!(
            rfs::read(rust_lib_name("first")),
            rfs::read(rust_lib_name("reproducible_build"))
        );
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .arg("-Zremap-path-prefix=")
            .arg("-Cdebuginfo=2")
            .run();
        rfs::copy(rust_lib_name("reproducible_build"), rust_lib_name("first"));
        rustc()
            .input("test/reproducible-build.rs")
            .crate_type("rlib")
            .arg("-Zremap-path-prefix=")
            .arg("-Cdebuginfo=2")
            .run();
        assert_eq!(
            rfs::read(rust_lib_name("first")),
            rfs::read(rust_lib_name("reproducible_build"))
        );
    });

    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .extern_("reproducible_build_aux", rust_lib_name("reproducible_build_aux"))
            .run();
        rfs::copy(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
        rfs::copy(rust_lib_name("reproducible_build_aux"), rust_lib_name("bar"));
        rustc()
            .input("reproducible-build.rs")
            .crate_type("rlib")
            .extern_("reproducible_build_aux", rust_lib_name("bar"))
            .run();
        assert_eq!(rfs::read(rust_lib_name("foo")), rfs::read(rust_lib_name("reproducible_build")));
    });
}
