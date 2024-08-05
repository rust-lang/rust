// This test checks that the `-C split-debuginfo` feature behaves in certain expected
// ways. It comes in 3 different flavours:
// * `off` - This indicates that split-debuginfo from the final artifact is
//   not desired. This is not supported on Windows and is the default on
//   Unix platforms except macOS. On macOS this means that `dsymutil` is
//   not executed.

// * `packed` - This means that debuginfo is desired in one location
//   separate from the main executable. This is the default on Windows
//   (`*.pdb`) and macOS (`*.dSYM`). On other Unix platforms this subsumes
//   `-Zsplit-dwarf=single` and produces a `*.dwp` file.

// * `unpacked` - This means that debuginfo will be roughly equivalent to
//   object files, meaning that it's throughout the build directory
//   rather than in one location (often the fastest for local development).
//   This is not the default on any platform and is not supported on Windows.
// For each test, compilation should be successful, output artifacts should be
// exactly what is expected, no more, no less, and in some cases, the dwarf dump
// of debuginfo should reveal a certain symbol and the current working directory,
// depending on if the scope passed was wrong or not.
// See https://github.com/rust-lang/rust/pull/81493

//@ ignore-windows
// Reason: Windows only supports packed debuginfo - nothing to test.
//@ ignore-riscv64
// Reason: on this platform only `-Csplit-debuginfo=off` is supported, see #120518

//FIXME(Oneirical): try it with ignore-cross-compile

use run_make_support::{
    bin_name, cwd, has_extension, is_darwin, llvm_dwarfdump, rfs, rust_lib_name, rustc,
    shallow_find_files, target, Rustc,
};

fn main() {
    if is_darwin() {
        // If disabled, don't run `dsymutil`.
        for dsym in shallow_find_files(cwd(), |path| has_extension(path, "dSYM")) {
            rfs::remove_file(dsym);
        }
        rustc().input("foo.rs").arg("-Cdebuginfo=2").arg("-Csplit-debuginfo=off").run();
        check_extension_not_exists("dSYM");

        // Packed by default, but only if debuginfo is requested
        for dsym in shallow_find_files(cwd(), |path| has_extension(path, "dSYM")) {
            rfs::remove_file(dsym);
        }
        rustc().input("foo.rs").run();
        check_extension_not_exists("dSYM");
        rustc().input("foo.rs").arg("-Cdebuginfo=2").run();
        check_extension_exists("dSYM");
        for dsym in shallow_find_files(cwd(), |path| has_extension(path, "dSYM")) {
            rfs::remove_file(dsym);
        }
        rustc().input("foo.rs").arg("-Cdebuginfo=2").arg("-Csplit-debuginfo=packed").run();
        check_extension_exists("dSYM");

        // Object files are preserved with unpacked and `dsymutil` isn't run
        for dsym in shallow_find_files(cwd(), |path| has_extension(path, "dSYM")) {
            rfs::remove_file(dsym);
        }
        rustc().input("foo.rs").arg("-Cdebuginfo=2").arg("-Csplit-debuginfo=unpacked").run();
        check_extension_exists("o");
        check_extension_not_exists("dSYM");
    } else {
        // PACKED

        // - Debuginfo in `.o` files
        // - `.o` deleted
        // - `.dwo` never created
        // - `.dwp` never created
        unstable_rustc().input("foo.rs").arg("-Cdebuginfo=2").arg("-Csplit-debuginfo=off").run();
        check_extension_not_exists("dwp");
        check_extension_not_exists("dwo");
        rustc().input("foo.rs").arg("-Cdebuginfo=2").run();
        check_extension_not_exists("dwp");
        check_extension_not_exists("dwo");

        let split_dwarf_types = [SplitDwarf::Split, SplitDwarf::Single];

        for dwarf_type in split_dwarf_types {
            // SPLIT
            // - Debuginfo in `.dwo` files
            // - `.o` deleted
            // - `.dwo` deleted
            // - `.dwp` present

            // SINGLE
            // - Debuginfo in `.o` files
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present
            run_test(DebugInfoTest {
                input: "foo.rs",
                extra_flag: None,
                crate_type: None,
                debug_info: DebugInfo::Packed,
                split_dwarf: dwarf_type,
                has_o: false,
                has_dwo: false,
                dwp: Some("foo"),
                output: Some(bin_name("foo")),
                remap_scope: None,
            });
        }

        for dwarf_type in split_dwarf_types {
            // SPLIT
            // - rmeta file added to rlib, no object files are generated and thus no debuginfo is
            // generated
            // - `.o` never created
            // - `.dwo` never created
            // - `.dwp` never created

            // SINGLE
            // - rmeta file added to rlib, no object files are generated and thus no debuginfo is
            // generated
            // - `.o` never created
            // - `.dwo` never created
            // - `.dwp` never created
            run_test(DebugInfoTest {
                input: "lto.rs",
                extra_flag: Some(ExtraFlag::LinkerPluginLto),
                crate_type: Some(CrateType::Rlib),
                debug_info: DebugInfo::Packed,
                split_dwarf: dwarf_type,
                has_o: false,
                has_dwo: false,
                dwp: None,
                output: Some(rust_lib_name("lto")),
                remap_scope: None,
            });
        }

        for dwarf_type in split_dwarf_types {
            // SPLIT
            // - Debuginfo in `.dwo` files
            // - `.o` and binary refer to remapped `.dwo` paths which do not exist
            // - `.o` deleted
            // - `.dwo` deleted
            // - `.dwp` present

            // SINGLE
            // - Debuginfo in `.o` files
            // - `.o` and binary refer to remapped `.o` paths which do not exist
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present

            run_test(DebugInfoTest {
                input: "foo.rs",
                extra_flag: None,
                crate_type: None,
                debug_info: DebugInfo::Packed,
                split_dwarf: dwarf_type,
                has_o: false,
                has_dwo: false,
                dwp: Some("foo"),
                output: Some(bin_name("foo")),
                remap_scope: Some(None),
            });
        }

        // - Debuginfo in `.o` files
        // - `.o` and binary refer to remapped `.o` paths which do not exist
        // - `.o` deleted
        // - `.dwo` never created
        // - `.dwp` present
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Packed,
            split_dwarf: SplitDwarf::Single,
            has_o: false,
            has_dwo: false,
            dwp: Some("foo"),
            output: Some(bin_name("foo")),
            remap_scope: Some(Some(Scope::Debug)),
        });

        // - Debuginfo in `.o` files
        // - `.o` and binary refer to remapped `.o` paths which do not exist
        // - `.o` deleted
        // - `.dwo` never created
        // - `.dwp` present
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Packed,
            split_dwarf: SplitDwarf::Single,
            has_o: false,
            has_dwo: false,
            dwp: Some("foo"),
            output: Some(bin_name("foo")),
            remap_scope: Some(Some(Scope::Macro)),
        });

        for dwarf_type in split_dwarf_types {
            // SPLIT
            // - Debuginfo in `.dwo` files
            // - (bar) `.rlib` file created, contains `.dwo`
            // - (bar) `.o` deleted
            // - (bar) `.dwo` deleted
            // - (bar) `.dwp` never created
            // - (main) `.o` deleted
            // - (main) `.dwo` deleted
            // - (main) `.dwp` present

            // SINGLE
            // - Debuginfo in `.o` files
            // - (bar) `.rlib` file created, contains `.o`
            // - (bar) `.o` deleted
            // - (bar) `.dwo` never created
            // - (bar) `.dwp` never created
            // - (main) `.o` deleted
            // - (main) `.dwo` never created
            // - (main) `.dwp` present

            run_test(DebugInfoTest {
                input: "bar.rs",
                extra_flag: None,
                crate_type: Some(CrateType::Lib),
                debug_info: DebugInfo::Packed,
                split_dwarf: dwarf_type,
                has_o: false,
                has_dwo: false,
                dwp: None,
                output: None,
                remap_scope: None,
            });
            run_test(DebugInfoTest {
                input: "main.rs",
                extra_flag: Some(ExtraFlag::ExternBar),
                crate_type: None,
                debug_info: DebugInfo::Packed,
                split_dwarf: dwarf_type,
                has_o: false,
                has_dwo: false,
                dwp: Some("main"),
                output: Some(bin_name("main")),
                remap_scope: None,
            });
        }

        // UNPACKED

        // - Debuginfo in `.dwo` files
        // - `.o` deleted
        // - `.dwo` present
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Split,
            has_o: false,
            has_dwo: true,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: None,
        });

        // - Debuginfo in `.o` files
        // - `.o` present
        // - `.dwo` never created
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: None,
        });

        for dwarf_type in split_dwarf_types {
            // SPLIT
            // - rmeta file added to rlib, no object files are generated and thus no debuginfo
            // is generated
            // - `.o` present (bitcode)
            // - `.dwo` never created
            // - `.dwp` never created

            // SINGLE
            // - rmeta file added to rlib, no object files are generated and thus no debuginfo
            // is generated
            // - `.o` present (bitcode)
            // - `.dwo` never created
            // - `.dwp` never created

            run_test(DebugInfoTest {
                input: "lto.rs",
                extra_flag: Some(ExtraFlag::LinkerPluginLto),
                crate_type: Some(CrateType::Rlib),
                debug_info: DebugInfo::Unpacked,
                split_dwarf: dwarf_type,
                has_o: true,
                has_dwo: false,
                dwp: None,
                output: Some(rust_lib_name("lto")),
                remap_scope: None,
            });
        }

        // - Debuginfo in `.dwo` files
        // - `.o` and binary refer to remapped `.dwo` paths which do not exist
        // - `.o` deleted
        // - `.dwo` present
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Split,
            has_o: false,
            has_dwo: true,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: Some(None),
        });

        // - Debuginfo in `.o` files
        // - `.o` and binary refer to remapped `.o` paths which do not exist
        // - `.o` present
        // - `.dwo` never created
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: Some(None),
        });

        // - Debuginfo in `.o` files
        // - `.o` and binary refer to remapped `.o` paths which do not exist
        // - `.o` present
        // - `.dwo` never created
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: Some(Some(Scope::Debug)),
        });

        // - Debuginfo in `.o` files
        // - `.o` and binary refer to remapped `.o` paths which do not exist
        // - `.o` present
        // - `.dwo` never created
        // - `.dwp` never created
        run_test(DebugInfoTest {
            input: "foo.rs",
            extra_flag: None,
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: Some(bin_name("foo")),
            remap_scope: Some(Some(Scope::Macro)),
        });

        // - Debuginfo in `.dwo` files
        // - (bar) `.rlib` file created, contains `.dwo`
        // - (bar) `.o` deleted
        // - (bar) `.dwo` present
        // - (bar) `.dwp` never created
        // - (main) `.o` deleted
        // - (main) `.dwo` present
        // - (main) `.dwp` never created
        run_test(DebugInfoTest {
            input: "bar.rs",
            extra_flag: None,
            crate_type: Some(CrateType::Lib),
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Split,
            has_o: false,
            has_dwo: true,
            dwp: None,
            output: None,
            remap_scope: None,
        });
        run_test(DebugInfoTest {
            input: "main.rs",
            extra_flag: Some(ExtraFlag::ExternBar),
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Split,
            has_o: false,
            has_dwo: true,
            dwp: None,
            output: Some(bin_name("main")),
            remap_scope: None,
        });

        // - Debuginfo in `.o` files
        // - (bar) `.rlib` file created, contains `.o`
        // - (bar) `.o` present
        // - (bar) `.dwo` never created
        // - (bar) `.dwp` never created
        // - (main) `.o` present
        // - (main) `.dwo` never created
        // - (main) `.dwp` never created
        run_test(DebugInfoTest {
            input: "bar.rs",
            extra_flag: None,
            crate_type: Some(CrateType::Lib),
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: None,
            remap_scope: None,
        });
        run_test(DebugInfoTest {
            input: "main.rs",
            extra_flag: Some(ExtraFlag::ExternBar),
            crate_type: None,
            debug_info: DebugInfo::Unpacked,
            split_dwarf: SplitDwarf::Single,
            has_o: true,
            has_dwo: false,
            dwp: None,
            output: Some(bin_name("main")),
            remap_scope: None,
        });
    }
}

#[track_caller]
fn run_test(params: DebugInfoTest) {
    let mut compiler = unstable_rustc();
    compiler.input(params.input).arg("-Cdebuginfo=2");
    match params.debug_info {
        DebugInfo::Packed => compiler.arg("-Csplit-debuginfo=packed"),
        DebugInfo::Unpacked => compiler.arg("-Csplit-debuginfo=unpacked"),
    };
    match params.split_dwarf {
        SplitDwarf::Split => compiler.arg("-Zsplit-dwarf-kind=split"),
        SplitDwarf::Single => compiler.arg("-Zsplit-dwarf-kind=single"),
    };
    if let Some(crate_type) = params.crate_type {
        match crate_type {
            CrateType::Rlib => compiler.crate_type("rlib"),
            CrateType::Lib => compiler.crate_type("lib"),
        };
    };

    if let Some(extra_flag) = params.extra_flag {
        match extra_flag {
            ExtraFlag::LinkerPluginLto => compiler.arg("-Clinker-plugin-lto"),
            ExtraFlag::ExternBar => compiler.extern_("bar", rust_lib_name("bar")),
        };
    };

    if let Some(remap) = params.remap_scope {
        compiler.remap_path_prefix(cwd(), "/a");
        if let Some(scope) = remap {
            match scope {
                Scope::Debug => compiler.arg("-Zremap-path-scope=debuginfo"),
                Scope::Macro => compiler.arg("-Zremap-path-scope=macro"),
            };
        };
    }
    compiler.run();
    if let Some(remap) = params.remap_scope {
        let mut objdump = llvm_dwarfdump();
        let out = objdump.arg("--debug-info").input("foo").run();
        out.assert_stdout_contains("DW_AT_GNU_dwo_name");
        if let Some(scope) = remap {
            match scope {
                Scope::Debug => out.assert_stdout_not_contains(cwd().display().to_string()),
                Scope::Macro => out.assert_stdout_contains(cwd().display().to_string()),
            };
        } else {
            out.assert_stdout_not_contains(cwd().display().to_string());
        };
    }
    if params.has_o {
        for object in shallow_find_files(cwd(), |path| has_extension(path, "o")) {
            rfs::remove_file(object);
        }
    } else {
        check_extension_not_exists("o");
    }
    if params.has_dwo {
        for dwo in shallow_find_files(cwd(), |path| has_extension(path, "dwo")) {
            rfs::remove_file(dwo);
        }
    } else {
        check_extension_not_exists("dwo");
    }
    if let Some(name) = params.dwp {
        rfs::remove_file(&format!("{name}.dwp"));
    } else {
        check_extension_not_exists("dwp");
    }
    if let Some(output) = params.output {
        rfs::remove_file(output);
    }
}

struct DebugInfoTest {
    input: &'static str,
    extra_flag: Option<ExtraFlag>,
    crate_type: Option<CrateType>,
    debug_info: DebugInfo,
    split_dwarf: SplitDwarf,
    has_o: bool,
    has_dwo: bool,
    dwp: Option<&'static str>,
    output: Option<String>,
    remap_scope: Option<Option<Scope>>,
}

enum CrateType {
    Lib,
    Rlib,
}

enum ExtraFlag {
    LinkerPluginLto,
    ExternBar,
}

#[derive(Clone, Copy)]
enum Scope {
    Debug,
    Macro,
}

enum DebugInfo {
    Packed,
    Unpacked,
}

#[derive(Clone, Copy)]
enum SplitDwarf {
    Split,
    Single,
}

// Some non-Windows, non-Darwin platforms are not stable, and some are.
fn unstable_rustc() -> Rustc {
    let mut compiler = rustc();
    if !target().contains("linux") {
        compiler.arg("-Zunstable-options");
    }
    compiler
}

#[track_caller]
fn check_extension_exists(ext: &str) {
    if shallow_find_files(cwd(), |path| has_extension(path, ext)).is_empty() {
        eprintln!("{:#?}", rfs::shallow_find_dir_entries(cwd()));
        panic!("a file with the requested extension {ext} was not found");
    }
}

#[track_caller]
fn check_extension_not_exists(ext: &str) {
    if !shallow_find_files(cwd(), |path| has_extension(path, ext)).is_empty() {
        eprintln!("{:#?}", rfs::shallow_find_dir_entries(cwd()));
        panic!("a file with the requested extension {ext} was unexpectedly found");
    }
}
