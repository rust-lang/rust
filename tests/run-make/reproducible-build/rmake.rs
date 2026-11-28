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
// Tracking Issue: https://github.com/rust-lang/rust/issues/129080

//@ ignore-cross-compile (linker binary needs to run)
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

use run_make_support::{
    bin_name, cwd, diff, is_darwin, is_windows, regex, rfs, run_in_tmpdir, rust_lib_name, rustc,
    target,
};

fn main() {
    // Smoke tests. Simple flags, build should be reproducible.
    eprintln!("smoke_test => None");
    smoke_test(None);
    eprintln!("smoke_test => SmokeFlag::Debug");
    smoke_test(Some(SmokeFlag::Debug));
    eprintln!("smoke_test => SmokeFlag::Opt");
    smoke_test(Some(SmokeFlag::Opt));

    // Builds should be reproducible even through custom library search paths
    // or remap path prefixes.
    eprintln!("paths_test => PathsFlag::Link");
    paths_test(PathsFlag::Link);
    eprintln!("paths_test => PathsFlag::Remap");
    paths_test(PathsFlag::Remap);

    // Builds should be reproducible even if each build is done in a different directory,
    // with both --remap-path-prefix and -Z remap-cwd-prefix.

    // FIXME: Cross-directory binary reproducibility checks are disabled on macOS and Windows.
    if !is_darwin() && !is_windows() {
        diff_dir_test(CrateType::Bin, RemapType::Path, 0);
        if target().contains("-linux-") {
            diff_dir_test(CrateType::Bin, RemapType::Path, 2);
        }
    }

    diff_dir_test(CrateType::Rlib, RemapType::Path, 0);
    if target().contains("-linux-") {
        diff_dir_test(CrateType::Rlib, RemapType::Path, 2);
    }

    // FIXME(#129117): Windows rlib builds with full debuginfo and
    // -Zremap-cwd-prefix are not reproducible.
    let debuginfo = if is_windows() { 0 } else { 2 };

    if !is_darwin() && !is_windows() {
        diff_dir_test(CrateType::Bin, RemapType::Cwd { is_empty: false }, debuginfo);
    }

    diff_dir_test(CrateType::Rlib, RemapType::Cwd { is_empty: false }, debuginfo);
    diff_dir_test(CrateType::Rlib, RemapType::Cwd { is_empty: true }, debuginfo);

    eprintln!("final extern test");
    // Builds should be reproducible when using the --extern flag.
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
        assert!(rfs::read(rust_lib_name("foo")) == rfs::read(rust_lib_name("reproducible_build")))
    });
}

#[track_caller]
fn smoke_test(flag: Option<SmokeFlag>) {
    run_in_tmpdir(|| {
        rustc().input("linker.rs").opt().run();
        rustc().input("reproducible-build-aux.rs").run();
        let mut compiler1 = rustc();
        let mut compiler2 = rustc();
        if let Some(flag) = flag {
            match flag {
                SmokeFlag::Debug => {
                    compiler1.arg("-g");
                    compiler2.arg("-g");
                }
                SmokeFlag::Opt => {
                    compiler1.opt();
                    compiler2.opt();
                }
            };
        };
        compiler1
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();
        compiler2
            .input("reproducible-build.rs")
            .linker(&cwd().join(bin_name("linker")).display().to_string())
            .run();

        #[cfg(not(target_os = "aix"))]
        {
            diff().actual_file("linker-arguments1").expected_file("linker-arguments2").run();
        }
        #[cfg(target_os = "aix")]
        {
            // The AIX link command includes an additional argument
            // that specifies the file containing exported symbols, e.g.,
            // -bE:/tmp/rustcO6hxkY/list.exp. In this example, the part of the
            // directory name "rustcO6hxkY" is randomly generated to ensure that
            // different linking processes do not collide. For the purpose
            // of comparing link arguments, the randomly generated part is
            // replaced with a placeholder.
            let content1 =
                std::fs::read_to_string("linker-arguments1").expect("Failed to read file");
            let content2 =
                std::fs::read_to_string("linker-arguments2").expect("Failed to read file");

            // Define the regex for the directory name containing the random substring.
            let re = regex::Regex::new(r"rustc[a-zA-Z0-9]{6}/list\.exp").expect("Invalid regex");

            // Compare link commands with random strings replaced by placeholders.
            assert!(
                re.replace_all(&content1, "rustcXXXXXX/list.exp").to_string()
                    == re.replace_all(&content2, "rustcXXXXXX/list.exp").to_string()
            );
        }
    });
}

#[track_caller]
fn paths_test(flag: PathsFlag) {
    run_in_tmpdir(|| {
        rustc().input("reproducible-build-aux.rs").run();
        let mut compiler1 = rustc();
        let mut compiler2 = rustc();
        match flag {
            PathsFlag::Link => {
                compiler1.library_search_path("a");
                compiler2.library_search_path("b");
            }
            PathsFlag::Remap => {
                compiler1.arg("--remap-path-prefix=/a=/c");
                compiler2.arg("--remap-path-prefix=/b=/c");
            }
        }
        compiler1.input("reproducible-build.rs").crate_type("rlib").run();
        rfs::rename(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
        compiler2.input("reproducible-build.rs").crate_type("rlib").run();
        assert!(rfs::read(rust_lib_name("foo")) == rfs::read(rust_lib_name("reproducible_build")))
    });
}

#[track_caller]
fn diff_dir_test(crate_type: CrateType, remap_type: RemapType, debuginfo: u32) {
    eprintln!("diff_dir_test => {crate_type:?}, {remap_type:?}, debuginfo={debuginfo}");
    run_in_tmpdir(|| {
        let base_dir = cwd();
        rustc().input("reproducible-build-aux.rs").run();
        rfs::create_dir("test");
        rfs::copy("reproducible-build.rs", "test/reproducible-build.rs");
        let mut compiler1 = rustc();
        let mut compiler2 = rustc();
        match crate_type {
            CrateType::Bin => {
                compiler1.crate_type("bin");
                compiler2.crate_type("bin");
            }
            CrateType::Rlib => {
                compiler1.crate_type("rlib");
                compiler2.crate_type("rlib");
            }
        }
        match remap_type {
            RemapType::Path => {
                compiler1.arg(&format!("--remap-path-prefix={}=/b", cwd().display()));
                compiler2
                    .arg(format!("--remap-path-prefix={}=/b", base_dir.join("test").display()));
            }
            RemapType::Cwd { is_empty } => {
                if is_empty {
                    compiler1.arg("-Zremap-cwd-prefix=");
                    compiler2.arg("-Zremap-cwd-prefix=");
                } else {
                    compiler1.arg("-Zremap-cwd-prefix=.");
                    compiler2.arg("-Zremap-cwd-prefix=.");
                }
            }
        }
        compiler1.arg(format!("-Cdebuginfo={debuginfo}"));
        compiler2.arg(format!("-Cdebuginfo={debuginfo}"));

        compiler1.input("reproducible-build.rs").run();
        match crate_type {
            CrateType::Bin => {
                rfs::rename(bin_name("reproducible-build"), bin_name("foo"));
            }
            CrateType::Rlib => {
                rfs::rename(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
            }
        }
        std::env::set_current_dir("test").unwrap();
        compiler2
            .input("reproducible-build.rs")
            .library_search_path(&base_dir)
            .out_dir(&base_dir)
            .run();
        std::env::set_current_dir(&base_dir).unwrap();
        match crate_type {
            CrateType::Bin => {
                #[cfg(not(target_os = "aix"))]
                {
                    assert!(
                        rfs::read(bin_name("reproducible-build")) == rfs::read(bin_name("foo"))
                    );
                }
                #[cfg(target_os = "aix")]
                {
                    // At the 4th-byte offset, the AIX XCOFF file header defines a
                    // 4-byte timestamp. Nullify the timestamp before performing a
                    // binary comparison.
                    let mut file1 = rfs::read(bin_name("reproducible-build"));
                    let mut file2 = rfs::read(bin_name("foo"));
                    assert!(file1[4..8].fill(0x00) == file2[4..8].fill(0x00));
                };
            }
            CrateType::Rlib => {
                assert!(
                    rfs::read(rust_lib_name("foo"))
                        == rfs::read(rust_lib_name("reproducible_build"))
                );
            }
        }
    });
}

enum SmokeFlag {
    Debug,
    Opt,
}

enum PathsFlag {
    Link,
    Remap,
}

#[derive(Debug)]
enum CrateType {
    Bin,
    Rlib,
}

#[derive(Debug)]
enum RemapType {
    Path,
    Cwd { is_empty: bool },
}
