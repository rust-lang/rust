// This test makes sure that --remap-path-prefix has the expected effects on paths in debuginfo.
// We explicitly switch to a directory that *is* a prefix of the directory our
// source code is contained in.
// It tests several cases, each of them has a detailed description attached to it.
// See https://github.com/rust-lang/rust/pull/96867

//@ needs-target-std
//@ ignore-windows
// Reason: the remap path prefix is not printed in the dwarf dump.

use run_make_support::{cwd, is_darwin, llvm_dwarfdump, rust_lib_name, rustc};

fn main() {
    // The compiler is called with an *ABSOLUTE PATH* as input, and that absolute path *is* within
    // the working directory of the compiler. We are remapping the path that contains `src`.
    check_dwarf(DwarfTest {
        lib_name: "abs_input_inside_working_dir",
        input_path: PathType::Absolute,
        scope: None,
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().display())),
        dwarf_test: DwarfDump::ContainsSrcPath,
    });
    check_dwarf(DwarfTest {
        lib_name: "abs_input_inside_working_dir_scope",
        input_path: PathType::Absolute,
        scope: Some(ScopeType::Object),
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().display())),
        dwarf_test: DwarfDump::ContainsSrcPath,
    });
    // The compiler is called with an *ABSOLUTE PATH* as input, and that absolute path is *not*
    // within the working directory of the compiler. We are remapping both the path that contains
    // `src` and the working directory to the same thing. This setup corresponds to a workaround
    // that is needed when trying to remap everything to something that looks like a local
    // path. Relative paths are interpreted as relative to the compiler's working directory (e.g.
    // in debuginfo). If we also remap the working directory, the compiler strip it from other
    // paths so that the final outcome is the desired one again.
    check_dwarf(DwarfTest {
        lib_name: "abs_input_outside_working_dir",
        input_path: PathType::Absolute,
        scope: None,
        remap_path_prefix: PrefixType::Dual((
            format!("{}=REMAPPED", cwd().display()),
            "rmake_out=REMAPPED".to_owned(),
        )),
        dwarf_test: DwarfDump::ContainsSrcPath,
    });
    // The compiler is called with a *RELATIVE PATH* as input. We are remapping the working
    // directory of the compiler, which naturally is an implicit prefix of our relative input path.
    // Debuginfo will expand the relative path to an absolute path and we expect the working
    // directory to be remapped in that expansion.
    check_dwarf(DwarfTest {
        lib_name: "rel_input_remap_working_dir",
        input_path: PathType::Relative,
        scope: None,
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().display())),
        dwarf_test: DwarfDump::ContainsSrcPath,
    });
    check_dwarf(DwarfTest {
        lib_name: "rel_input_remap_working_dir_scope",
        input_path: PathType::Relative,
        scope: Some(ScopeType::Object),
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().display())),
        dwarf_test: DwarfDump::ContainsSrcPath,
    });
    check_dwarf(DwarfTest {
        lib_name: "rel_input_remap_working_dir_scope",
        input_path: PathType::Relative,
        scope: Some(ScopeType::Diagnostics),
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().display())),
        dwarf_test: DwarfDump::AvoidSrcPath,
    });
    // The compiler is called with a *RELATIVE PATH* as input. We are remapping a *SUB-DIRECTORY*
    // of the compiler's working directory. This test makes sure that that directory is remapped
    // even though it won't actually show up in this form in the compiler's SourceMap and instead
    // is only constructed on demand during debuginfo generation.
    check_dwarf(DwarfTest {
        lib_name: "rel_input_remap_working_dir_child",
        input_path: PathType::Relative,
        scope: None,
        remap_path_prefix: PrefixType::Regular(format!("{}=REMAPPED", cwd().join("src").display())),
        dwarf_test: DwarfDump::ChildTest,
    });
    // The compiler is called with a *RELATIVE PATH* as input. We are remapping a
    // *PARENT DIRECTORY* of the compiler's working directory.
    check_dwarf(DwarfTest {
        lib_name: "rel_input_remap_working_dir_parent",
        input_path: PathType::Relative,
        scope: None,
        remap_path_prefix: PrefixType::Regular(format!(
            "{}=REMAPPED",
            cwd().parent().unwrap().display()
        )),
        dwarf_test: DwarfDump::ParentTest,
    });

    check_dwarf_deps("macro", DwarfDump::AvoidSrcPath);
    check_dwarf_deps("diagnostics", DwarfDump::AvoidSrcPath);
    check_dwarf_deps("macro,diagnostics", DwarfDump::AvoidSrcPath);
    check_dwarf_deps("object", DwarfDump::ContainsSrcPath);
}

#[track_caller]
fn check_dwarf_deps(scope: &str, dwarf_test: DwarfDump) {
    // build some_value.rs
    let mut rustc_sm = rustc();
    rustc_sm.input(cwd().join("src/some_value.rs"));
    rustc_sm.arg("-Cdebuginfo=2");
    rustc_sm.arg(format!("-Zremap-path-scope={}", scope));
    rustc_sm.arg("--remap-path-prefix");
    rustc_sm.arg(format!("{}=/REMAPPED", cwd().display()));
    rustc_sm.arg("-Csplit-debuginfo=off");
    rustc_sm.run();

    // build print_value.rs
    let print_value_rlib = rust_lib_name(&format!("print_value.{scope}"));
    let mut rustc_pv = rustc();
    rustc_pv.input(cwd().join("src/print_value.rs"));
    rustc_pv.output(&print_value_rlib);
    rustc_pv.arg("-Cdebuginfo=2");
    rustc_pv.arg(format!("-Zremap-path-scope={}", scope));
    rustc_pv.arg("--remap-path-prefix");
    rustc_pv.arg(format!("{}=/REMAPPED", cwd().display()));
    rustc_pv.arg("-Csplit-debuginfo=off");
    rustc_pv.run();

    match dwarf_test {
        DwarfDump::AvoidSrcPath => {
            llvm_dwarfdump()
                .input(print_value_rlib)
                .run()
                .assert_stdout_not_contains("REMAPPED/src/some_value.rs")
                .assert_stdout_not_contains("REMAPPED/src/print_value.rs")
                .assert_stdout_not_contains("REMAPPED/REMAPPED")
                .assert_stdout_contains(cwd().join("src/some_value.rs").display().to_string())
                .assert_stdout_contains(cwd().join("src/print_value.rs").display().to_string());
        }
        DwarfDump::ContainsSrcPath => {
            llvm_dwarfdump()
                .input(print_value_rlib)
                .run()
                .assert_stdout_contains("REMAPPED/src/some_value.rs")
                .assert_stdout_contains("REMAPPED/src/print_value.rs")
                .assert_stdout_not_contains(cwd().join("src/some_value.rs").display().to_string())
                .assert_stdout_not_contains(cwd().join("src/print_value.rs").display().to_string());
        }
        _ => unreachable!(),
    }
}

#[track_caller]
fn check_dwarf(test: DwarfTest) {
    let mut rustc = rustc();
    match test.input_path {
        PathType::Absolute => rustc.input(cwd().join("src/quux.rs")),
        PathType::Relative => rustc.input("src/quux.rs"),
    };
    rustc.output(rust_lib_name(test.lib_name));
    rustc.arg("-Cdebuginfo=2");
    if let Some(scope) = test.scope {
        match scope {
            ScopeType::Object => rustc.arg("-Zremap-path-scope=object"),
            ScopeType::Diagnostics => rustc.arg("-Zremap-path-scope=diagnostics"),
        };
        if is_darwin() {
            rustc.arg("-Csplit-debuginfo=off");
        }
    }
    match test.remap_path_prefix {
        PrefixType::Regular(prefix) => {
            // We explicitly switch to a directory that *is* a prefix of the directory our
            // source code is contained in.
            rustc.arg("--remap-path-prefix");
            rustc.arg(prefix);
        }
        PrefixType::Dual((prefix1, prefix2)) => {
            // We explicitly switch to a directory that is *not* a prefix of the directory our
            // source code is contained in.
            rustc.arg("--remap-path-prefix");
            rustc.arg(prefix1);
            rustc.arg("--remap-path-prefix");
            rustc.arg(prefix2);
        }
    }
    rustc.run();
    match test.dwarf_test {
        DwarfDump::ContainsSrcPath => {
            llvm_dwarfdump()
                .input(rust_lib_name(test.lib_name))
                .run()
                // We expect the path to the main source file to be remapped.
                .assert_stdout_contains("REMAPPED/src/quux.rs")
                // No weird duplication of remapped components (see #78479)
                .assert_stdout_not_contains("REMAPPED/REMAPPED");
        }
        DwarfDump::AvoidSrcPath => {
            llvm_dwarfdump()
                .input(rust_lib_name(test.lib_name))
                .run()
                .assert_stdout_not_contains("REMAPPED/src/quux.rs")
                .assert_stdout_not_contains("REMAPPED/REMAPPED");
        }
        DwarfDump::ChildTest => {
            llvm_dwarfdump()
                .input(rust_lib_name(test.lib_name))
                .run()
                // We expect `src/quux.rs` to have been remapped to `REMAPPED/quux.rs`.
                .assert_stdout_contains("REMAPPED/quux.rs")
                // We don't want to find the path that we just remapped anywhere in the DWARF
                .assert_stdout_not_contains(cwd().join("src").to_str().unwrap())
                // No weird duplication of remapped components (see #78479)
                .assert_stdout_not_contains("REMAPPED/REMAPPED");
        }
        DwarfDump::ParentTest => {
            llvm_dwarfdump()
                .input(rust_lib_name(test.lib_name))
                .run()
                // We expect `src/quux.rs` to have been remapped to
                // `REMAPPED/remap-path-prefix-dwarf/src/quux.rs`.
                .assert_stdout_contains("REMAPPED/rmake_out/src/quux.rs")
                // We don't want to find the path that we just remapped anywhere in the DWARF
                .assert_stdout_not_contains(cwd().parent().unwrap().to_str().unwrap())
                // No weird duplication of remapped components (see #78479)
                .assert_stdout_not_contains("REMAPPED/REMAPPED");
        }
    };
}

struct DwarfTest {
    lib_name: &'static str,
    input_path: PathType,
    scope: Option<ScopeType>,
    remap_path_prefix: PrefixType,
    dwarf_test: DwarfDump,
}

enum PathType {
    Absolute,
    Relative,
}

enum ScopeType {
    Object,
    Diagnostics,
}

enum DwarfDump {
    ContainsSrcPath,
    AvoidSrcPath,
    ChildTest,
    ParentTest,
}

enum PrefixType {
    Regular(String),
    Dual((String, String)),
}
