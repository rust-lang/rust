// ignore-tidy-linelength
//! Basic smoke tests for behavior of `-C split-debuginfo` and the combined behavior when used in
//! conjunction with other flags such as:
//!
//! - `--remap-path-prefix`: see
//!   <https://doc.rust-lang.org/rustc/command-line-arguments.html#--remap-path-prefix-remap-source-names-in-output>.
//! - `-Z remap-path-scope`: see
//!     - <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/remap-path-scope.html>
//!     - <https://github.com/rust-lang/rust/issues/111540>
//!     - RFC #3127 trim-paths: <https://github.com/rust-lang/rfcs/pull/3127>
//! - `-Z split-dwarf-kind`: see <https://github.com/rust-lang/rust/pull/89819>.
//! - `-Clinker-plugin-lto`: see <https://doc.rust-lang.org/rustc/linker-plugin-lto.html>.
//!
//! # Test implementation remark
//!
//! - The pattern match on enum variants are intentional, because I find that they are very
//!   revealing with respect to the kind of test coverage that we have and don't have.
//!
//! # Known limitations
//!
//! - The linux test coverage of cross-interactions between `-C split-debuginfo` and other flags are
//!   significantly higher than the lack of such coverage for Windows and Darwin.
//! - windows-gnu is not tested at all, see the `FIXME(#135531)`s below.
//! - This test for the most part merely checks for existence/absence of certain artifacts, it does
//!   not sanity check if the debuginfo artifacts are actually usable or contains the expected
//!   amount/quality of debuginfo, especially on windows-msvc and darwin.
//! - FIXME(#111540): this test has insufficient coverage in relation to trim-paths RFC, see also
//!   the comment <https://github.com/rust-lang/rust/issues/111540#issuecomment-1994010274>. The
//!   basic `llvm-dwarfdump` textual output inspection here is very fragile. The original `Makefile`
//!   version used `objdump` (not to be confused with `llvm-objdump`) but inspected the wrong line
//!   because it was looking at `DW_AT_GNU_dwo_name` when it should've been looking at
//!   `DW_AT_comp_dir`.
//! - This test does not have good coverage for what values of `-Csplit-debuginfo` are stable vs
//!   non-stable for the various targets, i.e. which values *should* be gated behind
//!   `-Zunstable-options` for a given target. The `Makefile` version yolo'd a `-Zunstable-options`
//!   for non-windows + non-linux + non-darwin, but had a misplaced interpolation which suggested to
//!   me that that conditional `-Zunstable-options` never actually materialized.
//!
//! # Additional references
//!
//! - Apple `.dSYM` debug symbol bundles: <https://lldb.llvm.org/use/symbols.html>.
//! - LLVM `dsymutil`: <https://llvm.org/docs/CommandGuide/dsymutil.html>.

// NOTE: this is a host test
//@ ignore-cross-compile

// NOTE: this seems to be a host test, and testing on host `riscv64-gc-unknown-linux-gnu` reveals
// that this test is failing because of [MC: "error: A dwo section may not contain relocations" when
// building with fission + RISCV64 #56642](https://github.com/llvm/llvm-project/issues/56642). This
// test is ignored for now to unblock efforts to bring riscv64 targets to be exercised in CI, cf.
// [Enable riscv64gc-gnu testing #126641](https://github.com/rust-lang/rust/pull/126641).
//@ ignore-riscv64 (https://github.com/llvm/llvm-project/issues/56642)

// FIXME(#135531): the `Makefile` version practically didn't test `-C split-debuginfo` on Windows
// at all, and lumped windows-msvc and windows-gnu together at that.
//@ ignore-windows-gnu

#![deny(warnings)]

use std::collections::BTreeSet;

use run_make_support::rustc::Rustc;
use run_make_support::{
    cwd, has_extension, is_darwin, is_windows, is_windows_msvc, llvm_dwarfdump, run_in_tmpdir,
    rustc, shallow_find_directories, shallow_find_files, uname,
};

/// `-C debuginfo`. See <https://doc.rust-lang.org/rustc/codegen-options/index.html#debuginfo>.
#[derive(Debug, PartialEq, Copy, Clone)]
enum DebuginfoLevel {
    /// `-C debuginfo=0` or `-C debuginfo=none` aka no debuginfo at all, default.
    None,
    /// `-C debuginfo=2` aka full debuginfo, aliased via `-g`.
    Full,
    /// The cli flag is not explicitly provided; default.
    Unspecified,
}

impl DebuginfoLevel {
    fn cli_value(&self) -> &'static str {
        // `-Cdebuginfo=...`
        match self {
            DebuginfoLevel::None => "none",
            DebuginfoLevel::Full => "2",
            DebuginfoLevel::Unspecified => unreachable!(),
        }
    }
}

/// `-C split-debuginfo`. See
/// <https://doc.rust-lang.org/rustc/codegen-options/index.html#split-debuginfo>.
///
/// Note that all three options are supported on Linux and Apple platforms, packed is supported on
/// Windows-MSVC, and all other platforms support off. Attempting to use an unsupported option
/// requires using the nightly channel with the `-Z unstable-options` flag.
#[derive(Debug, PartialEq, Copy, Clone)]
enum SplitDebuginfo {
    /// `-C split-debuginfo=off`. Default for platforms with ELF binaries and windows-gnu (not
    /// Windows MSVC and not macOS). Typically DWARF debug information can be found in the final
    /// artifact in sections of the executable.
    ///
    /// - Not supported on Windows MSVC.
    /// - On macOS this options prevents the final execution of `dsymutil` to generate debuginfo.
    Off,
    /// `-C split-debuginfo=unpacked`. Debug information will be found in separate files for each
    /// compilation unit (object file).
    ///
    /// - Not supported on Windows MSVC.
    /// - On macOS this means the original object files will contain debug information.
    /// - On other Unix platforms this means that `*.dwo` files will contain debug information.
    Unpacked,
    /// `-C split-debuginfo=packed`. Default for Windows MSVC and macOS. "Packed" here means that
    /// all the debug information is packed into a separate file from the main executable.
    ///
    /// - On Windows MSVC this is a `*.pdb` file.
    /// - On macOS this is a `*.dSYM` folder.
    /// - On other platforms this is a `*.dwp` file.
    Packed,
    /// The cli flag is not explicitly provided; uses platform default.
    Unspecified,
}

impl SplitDebuginfo {
    fn cli_value(&self) -> &'static str {
        // `-Csplit-debuginfo=...`
        match self {
            SplitDebuginfo::Off => "off",
            SplitDebuginfo::Unpacked => "unpacked",
            SplitDebuginfo::Packed => "packed",
            SplitDebuginfo::Unspecified => unreachable!(),
        }
    }
}

/// `-Z split-dwarf-kind`
#[derive(Debug, PartialEq, Copy, Clone)]
enum SplitDwarfKind {
    /// `-Zsplit-dwarf-kind=split`
    Split,
    /// `-Zsplit-dwarf-kind=single`
    Single,
    Unspecified,
}

impl SplitDwarfKind {
    fn cli_value(&self) -> &'static str {
        // `-Zsplit-dwarf-kind=...`
        match self {
            SplitDwarfKind::Split => "split",
            SplitDwarfKind::Single => "single",
            SplitDwarfKind::Unspecified => unreachable!(),
        }
    }
}

/// `-C linker-plugin-lto`
#[derive(Debug, PartialEq, Copy, Clone)]
enum LinkerPluginLto {
    /// Pass `-C linker-plugin-lto`.
    Yes,
    /// Don't pass `-C linker-plugin-lto`.
    Unspecified,
}

/// `--remap-path-prefix` or not.
#[derive(Debug, Clone)]
enum RemapPathPrefix {
    /// `--remap-path-prefix=$prefix=$remapped_prefix`.
    Yes { remapped_prefix: &'static str },
    /// Don't pass `--remap-path-prefix`.
    Unspecified,
}

/// `-Zremap-path-scope`. See
/// <https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/remap-path-scope.html#remap-path-scope>.
#[derive(Debug, Clone)]
enum RemapPathScope {
    /// Comma-separated list of remap scopes: `macro`, `diagnostics`, `debuginfo`, `object`, `all`.
    Yes(&'static str),
    Unspecified,
}

/// Whether to pass `-Zunstable-options`.
#[derive(Debug, PartialEq, Copy, Clone)]
enum UnstableOptions {
    Yes,
    Unspecified,
}

#[track_caller]
fn dwo_out_filenames(dwo_out: Option<&str>) -> BTreeSet<String> {
    let dwo_out = if let Some(d) = dwo_out {
        d
    } else {
        return BTreeSet::new();
    };
    let files = shallow_find_files(dwo_out, |path| {
        // Fiilter out source files
        !has_extension(path, "rs")
    });
    files
        .iter()
        .map(|p| {
            format!("{}/{}", dwo_out, p.file_name().unwrap().to_os_string().into_string().unwrap())
        })
        .collect()
}

#[track_caller]
fn cwd_filenames() -> BTreeSet<String> {
    let files = shallow_find_files(cwd(), |path| {
        // Fiilter out source files
        !has_extension(path, "rs")
    });
    files.iter().map(|p| p.file_name().unwrap().to_os_string().into_string().unwrap()).collect()
}

#[track_caller]
fn dwo_out_dwo_filenames(dwo_out: &str) -> BTreeSet<String> {
    let files = shallow_find_files(dwo_out, |p| has_extension(p, "dwo"));
    files
        .iter()
        .map(|p| {
            format!("{}/{}", dwo_out, p.file_name().unwrap().to_os_string().into_string().unwrap())
        })
        .collect()
}

#[track_caller]
fn cwd_dwo_filenames() -> BTreeSet<String> {
    let files = shallow_find_files(cwd(), |path| has_extension(path, "dwo"));
    files.iter().map(|p| p.file_name().unwrap().to_os_string().into_string().unwrap()).collect()
}

#[track_caller]
fn cwd_object_filenames() -> BTreeSet<String> {
    let files = shallow_find_files(cwd(), |path| has_extension(path, "o"));
    files.iter().map(|p| p.file_name().unwrap().to_os_string().into_string().unwrap()).collect()
}

#[must_use]
struct FileAssertions<'expected> {
    expected_files: BTreeSet<&'expected str>,
}

impl<'expected> FileAssertions<'expected> {
    #[track_caller]
    fn assert_on(self, found_files: &BTreeSet<String>) {
        let found_files: BTreeSet<_> = found_files.iter().map(|f| f.as_str()).collect();
        assert!(
            found_files.is_superset(&self.expected_files),
            "expected {:?} to exist, but only found {:?}",
            self.expected_files,
            found_files
        );

        let unexpected_files: BTreeSet<_> =
            found_files.difference(&self.expected_files).copied().collect();
        assert!(unexpected_files.is_empty(), "found unexpected files: {:?}", unexpected_files);
    }
}

/// Windows MSVC only supports packed debuginfo.
mod windows_msvc_tests {
    use super::*;

    pub(crate) fn split_debuginfo(split_kind: SplitDebuginfo, level: DebuginfoLevel) {
        // NOTE: `-C debuginfo` and other flags are not exercised here on Windows MSVC.
        run_in_tmpdir(|| {
            println!("checking: split_kind={:?} + level={:?}", split_kind, level);
            match (split_kind, level) {
                (SplitDebuginfo::Off, _) => {
                    rustc()
                        .input("foo.rs")
                        .split_debuginfo(split_kind.cli_value())
                        .run_fail()
                        .assert_stderr_contains(
                            "error: `-Csplit-debuginfo=off` is unstable on this platform",
                        );
                }
                (SplitDebuginfo::Unpacked, _) => {
                    rustc()
                        .input("foo.rs")
                        .split_debuginfo(split_kind.cli_value())
                        .run_fail()
                        .assert_stderr_contains(
                            "error: `-Csplit-debuginfo=unpacked` is unstable on this platform",
                        );
                }
                (SplitDebuginfo::Packed, _) => {
                    rustc().input("foo.rs").split_debuginfo(split_kind.cli_value()).run();

                    let found_files = cwd_filenames();
                    FileAssertions { expected_files: BTreeSet::from(["foo.exe", "foo.pdb"]) }
                        .assert_on(&found_files);
                }
                (split_kind, level) => {
                    panic!(
                        "split_kind={:?} + level={:?} is not handled on Windows MSVC",
                        split_kind, level
                    )
                }
            }
        });
    }
}

mod darwin_tests {
    use super::*;

    pub(crate) fn split_debuginfo(split_kind: SplitDebuginfo, level: DebuginfoLevel) {
        run_in_tmpdir(|| {
            println!("checking: split_kind={:?} + level={:?}", split_kind, level);

            let dsym_directories =
                || shallow_find_directories(cwd(), |path| has_extension(path, "dSYM"));

            match (split_kind, level) {
                (_, DebuginfoLevel::Unspecified) => {
                    rustc().input("foo.rs").run();
                    let directories =
                        shallow_find_directories(cwd(), |path| has_extension(path, "dSYM"));
                    assert!(
                        directories.is_empty(),
                        "expected no `*.dSYM` folder to be generated when `-Cdebuginfo` is not specified"
                    );
                }
                (_, DebuginfoLevel::None) => {
                    rustc().input("foo.rs").debuginfo(level.cli_value()).run();
                    let directories = dsym_directories();
                    assert!(
                        directories.is_empty(),
                        "expected no `*.dSYM` folder to be generated when `-Cdebuginfo=none`"
                    );
                }
                (SplitDebuginfo::Off, _) => {
                    rustc()
                        .input("foo.rs")
                        .split_debuginfo(split_kind.cli_value())
                        .debuginfo(level.cli_value())
                        .run();
                    let directories = dsym_directories();
                    assert!(
                        directories.is_empty(),
                        "expected no `*.dSYM` folder to be generated since we expect `-Csplit-debuginfo=off` to inhibit final debuginfo generation on macOS"
                    );
                }
                (SplitDebuginfo::Unpacked, _) => {
                    rustc().input("foo.rs").split_debuginfo(split_kind.cli_value()).run();
                    let directories = dsym_directories();
                    assert!(
                        directories.is_empty(),
                        "expected no `*.dSYM` folder to be generated since we expect on macOS the object files to contain debuginfo instead"
                    );
                }
                (SplitDebuginfo::Packed, DebuginfoLevel::Full) => {
                    rustc()
                        .input("foo.rs")
                        .split_debuginfo(split_kind.cli_value())
                        .debuginfo(level.cli_value())
                        .run();
                    let directories = shallow_find_directories(cwd(), |path| {
                        path.file_name().unwrap() == "foo.dSYM"
                    });
                    assert_eq!(directories.len(), 1, "failed to find `foo.dSYM`");
                }
                (SplitDebuginfo::Unspecified, DebuginfoLevel::Full) => {
                    rustc().input("foo.rs").debuginfo(level.cli_value()).run();
                    let directories = shallow_find_directories(cwd(), |path| {
                        path.file_name().unwrap() == "foo.dSYM"
                    });
                    assert_eq!(directories.len(), 1, "failed to find `foo.dSYM`");
                }
            }
        });
    }
}

mod shared_linux_other_tests {
    use std::path::PathBuf;

    use super::*;

    fn rustc(unstable_options: UnstableOptions) -> Rustc {
        if unstable_options == UnstableOptions::Yes {
            let mut rustc = run_make_support::rustc();
            rustc.arg("-Zunstable-options");
            rustc
        } else {
            run_make_support::rustc()
        }
    }

    #[derive(PartialEq)]
    pub(crate) enum CrossCrateTest {
        Yes,
        No,
    }

    pub(crate) fn split_debuginfo(
        cross_crate_test: CrossCrateTest,
        unstable_options: UnstableOptions,
        split_kind: SplitDebuginfo,
        level: DebuginfoLevel,
        split_dwarf_kind: SplitDwarfKind,
        lto: LinkerPluginLto,
        remap_path_prefix: RemapPathPrefix,
        remap_path_scope: RemapPathScope,
        split_dwarf_output_directory: Option<&str>,
    ) {
        run_in_tmpdir(|| {
            println!(
                "checking: unstable_options={:?} + split_kind={:?} + level={:?} + split_dwarf_kind={:?} + lto={:?} + remap_path_prefix={:?} + remap_path_scope={:?} + split_dwarf_out_dir={:?}",
                unstable_options,
                split_kind,
                level,
                split_dwarf_kind,
                lto,
                remap_path_prefix,
                remap_path_scope,
                split_dwarf_output_directory,
            );

            match cross_crate_test {
                CrossCrateTest::Yes => cross_crate_split_debuginfo(
                    unstable_options,
                    split_kind,
                    level,
                    split_dwarf_kind,
                    lto,
                    remap_path_prefix,
                    remap_path_scope,
                    split_dwarf_output_directory,
                ),
                CrossCrateTest::No => simple_split_debuginfo(
                    unstable_options,
                    split_kind,
                    level,
                    split_dwarf_kind,
                    lto,
                    remap_path_prefix,
                    remap_path_scope,
                    split_dwarf_output_directory,
                ),
            }
        });
    }

    fn cross_crate_split_debuginfo(
        unstable_options: UnstableOptions,
        split_kind: SplitDebuginfo,
        level: DebuginfoLevel,
        split_dwarf_kind: SplitDwarfKind,
        lto: LinkerPluginLto,
        remap_path_prefix: RemapPathPrefix,
        remap_path_scope: RemapPathScope,
        split_dwarf_output_directory: Option<&str>,
    ) {
        if let Some(dwo_out) = split_dwarf_output_directory {
            run_make_support::rfs::create_dir(dwo_out);
        }
        match (split_kind, level, split_dwarf_kind, lto, remap_path_prefix, remap_path_scope) {
            // packed-crosscrate-split
            // - Debuginfo in `.dwo` files
            // - (bar) `.rlib` file created, contains `.dwo`
            // - (bar) `.o` deleted
            // - (bar) `.dwo` deleted
            // - (bar) `.dwp` never created
            // - (main) `.o` deleted
            // - (main) `.dwo` deleted
            // - (main) `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("bar.rs")
                    .crate_type("lib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["libbar.rlib"]) }
                    .assert_on(&found_files);

                rustc(unstable_options)
                    .extern_("bar", "libbar.rlib")
                    .input("main.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let found_files = cwd_filenames();
                FileAssertions {
                    expected_files: BTreeSet::from(["libbar.rlib", "main", "main.dwp"]),
                }
                .assert_on(&found_files);
            }

            // packed-crosscrate-single
            // - Debuginfo in `.o` files
            // - (bar) `.rlib` file created, contains `.o`
            // - (bar) `.o` deleted
            // - (bar) `.dwo` never created
            // - (bar) `.dwp` never created
            // - (main) `.o` deleted
            // - (main) `.dwo` never created
            // - (main) `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("bar.rs")
                    .crate_type("lib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["libbar.rlib"]) }
                    .assert_on(&found_files);

                rustc(unstable_options)
                    .extern_("bar", "libbar.rlib")
                    .input("main.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let found_files = cwd_filenames();
                FileAssertions {
                    expected_files: BTreeSet::from(["libbar.rlib", "main", "main.dwp"]),
                }
                .assert_on(&found_files);
            }

            // unpacked-crosscrate-split
            // - Debuginfo in `.dwo` files
            // - (bar) `.rlib` file created, contains `.dwo`
            // - (bar) `.o` deleted
            // - (bar) `.dwo` present
            // - (bar) `.dwp` never created
            // - (main) `.o` deleted
            // - (main) `.dwo` present
            // - (main) `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("bar.rs")
                    .crate_type("lib")
                    .split_debuginfo(split_kind.cli_value())
                    .split_dwarf_out_dir(split_dwarf_output_directory)
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let mut bar_found_files = cwd_filenames();
                bar_found_files.append(&mut dwo_out_filenames(split_dwarf_output_directory));

                let bar_dwo_files = if let Some(dwo_out) = split_dwarf_output_directory {
                    dwo_out_dwo_filenames(dwo_out)
                } else {
                    cwd_dwo_filenames()
                };
                assert_eq!(bar_dwo_files.len(), 1);

                let mut bar_expected_files = BTreeSet::new();
                bar_expected_files.extend(bar_dwo_files);
                bar_expected_files.insert("libbar.rlib".to_string());

                FileAssertions {
                    expected_files: bar_expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&bar_found_files);

                rustc(unstable_options)
                    .extern_("bar", "libbar.rlib")
                    .input("main.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .split_dwarf_out_dir(split_dwarf_output_directory)
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let mut overall_found_files = cwd_filenames();
                overall_found_files.append(&mut dwo_out_filenames(split_dwarf_output_directory));

                let overall_dwo_files = if let Some(dwo_out) = split_dwarf_output_directory {
                    dwo_out_dwo_filenames(dwo_out)
                } else {
                    cwd_dwo_filenames()
                };
                assert_eq!(overall_dwo_files.len(), 2);

                let mut overall_expected_files = BTreeSet::new();
                overall_expected_files.extend(overall_dwo_files);
                overall_expected_files.insert("main".to_string());
                overall_expected_files.insert("libbar.rlib".to_string());

                FileAssertions {
                    expected_files: overall_expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&overall_found_files);
            }

            // unpacked-crosscrate-single
            // - Debuginfo in `.o` files
            // - (bar) `.rlib` file created, contains `.o`
            // - (bar) `.o` present
            // - (bar) `.dwo` never created
            // - (bar) `.dwp` never created
            // - (main) `.o` present
            // - (main) `.dwo` never created
            // - (main) `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("bar.rs")
                    .crate_type("lib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let bar_found_files = cwd_filenames();

                let bar_object_files = cwd_object_filenames();
                assert_eq!(bar_object_files.len(), 1);

                let mut bar_expected_files = BTreeSet::new();
                bar_expected_files.extend(bar_object_files);
                bar_expected_files.insert("libbar.rlib".to_string());

                FileAssertions {
                    expected_files: bar_expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&bar_found_files);

                rustc(unstable_options)
                    .extern_("bar", "libbar.rlib")
                    .input("main.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();

                let overall_found_files = cwd_filenames();

                let overall_object_files = cwd_object_filenames();
                assert_eq!(overall_object_files.len(), 2);

                let mut overall_expected_files = BTreeSet::new();
                overall_expected_files.extend(overall_object_files);
                overall_expected_files.insert("main".to_string());
                overall_expected_files.insert("libbar.rlib".to_string());

                FileAssertions {
                    expected_files: overall_expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&overall_found_files);
            }

            _ => {}
        }
    }

    fn simple_split_debuginfo(
        unstable_options: UnstableOptions,
        split_kind: SplitDebuginfo,
        level: DebuginfoLevel,
        split_dwarf_kind: SplitDwarfKind,
        lto: LinkerPluginLto,
        remap_path_prefix: RemapPathPrefix,
        remap_path_scope: RemapPathScope,
        split_dwarf_output_directory: Option<&str>,
    ) {
        if let Some(dwo_out) = split_dwarf_output_directory {
            run_make_support::rfs::create_dir(dwo_out);
        }
        match (split_kind, level, split_dwarf_kind, lto, remap_path_prefix, remap_path_scope) {
            // off (unspecified):
            // - Debuginfo in `.o` files
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unspecified,
                DebuginfoLevel::Full,
                SplitDwarfKind::Unspecified,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options).input("foo.rs").debuginfo(level.cli_value()).run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo"]) }.assert_on(&found_files);
            }

            // off:
            // - Debuginfo in `.o` files
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Off,
                DebuginfoLevel::Full,
                SplitDwarfKind::Unspecified,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo"]) }.assert_on(&found_files);
            }

            // packed-split:
            // - Debuginfo in `.dwo` files
            // - `.o` deleted
            // - `.dwo` deleted
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);
            }

            // packed-single:
            // - Debuginfo in `.o` files
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);
            }

            // packed-lto-split::
            // - `rmeta` file added to `rlib`, no object files are generated and thus no
            //   debuginfo is generated.
            // - `.o` never created
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Yes,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("baz.rs")
                    .crate_type("rlib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .arg("-Clinker-plugin-lto")
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["libbaz.rlib"]) }
                    .assert_on(&found_files);
            }

            // packed-lto-single:
            // - `rmeta` file added to `rlib`, no object files are generated and thus no
            //   debuginfo is generated
            // - `.o` never created
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Yes,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("baz.rs")
                    .crate_type("rlib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .arg("-Clinker-plugin-lto")
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["libbaz.rlib"]) }
                    .assert_on(&found_files);
            }

            // packed-remapped-split:
            // - Debuginfo in `.dwo` files
            // - `.o` and binary refer to remapped `.dwo` paths which do not exist
            // - `.o` deleted
            // - `.dwo` deleted
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // packed-remapped-single:
            // - `.o` and binary refer to remapped `.o` paths which do not exist
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // packed-remapped-scope:
            // - Debuginfo in `.o` files
            // - `.o` and binary refer to remapped `.o` paths which do not exist
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Yes(scope @ "debuginfo"),
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .arg(format!("-Zremap-path-scope={scope}"))
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // packed-remapped-wrong-scope:
            // - `.o` and binary refer to un-remapped `.o` paths because remap path scope is
            //   macro.
            // - `.o` deleted
            // - `.dwo` never created
            // - `.dwp` present
            (
                SplitDebuginfo::Packed,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Yes(scope @ "macro"),
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .arg(format!("-Zremap-path-scope={scope}"))
                    .run();
                let found_files = cwd_filenames();
                FileAssertions { expected_files: BTreeSet::from(["foo", "foo.dwp"]) }
                    .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::NoRemap);
            }

            // unpacked-split
            // - Debuginfo in `.dwo` files
            // - `.o` deleted
            // - `.dwo` present
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .split_dwarf_out_dir(split_dwarf_output_directory)
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();
                let mut found_files = cwd_filenames();
                found_files.append(&mut dwo_out_filenames(split_dwarf_output_directory));

                let dwo_files = if let Some(dwo_dir) = split_dwarf_output_directory {
                    dwo_out_dwo_filenames(dwo_dir)
                } else {
                    cwd_dwo_filenames()
                };
                assert_eq!(dwo_files.len(), 1);
                let mut expected_files = BTreeSet::new();
                expected_files.extend(dwo_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_str).collect(),
                }
                .assert_on(&found_files);
            }

            // unpacked-single
            // - Debuginfo in `.o` files
            // - `.o` present
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .run();
                let found_files = cwd_filenames();

                let object_files = cwd_object_filenames();
                assert_eq!(object_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(object_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_str).collect(),
                }
                .assert_on(&found_files);
            }

            // unpacked-lto-split
            // - `rmeta` file added to `rlib`, no object files are generated and thus no debuginfo
            //   is generated
            // - `.o` not present
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Yes,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("baz.rs")
                    .crate_type("rlib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .arg("-Clinker-plugin-lto")
                    .run();

                let found_files = cwd_filenames();

                FileAssertions { expected_files: BTreeSet::from(["libbaz.rlib"]) }
                    .assert_on(&found_files);
            }

            // unpacked-lto-single
            // - rmeta file added to rlib, no object files are generated and thus no debuginfo is generated
            // - `.o` present (bitcode)
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Yes,
                RemapPathPrefix::Unspecified,
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("baz.rs")
                    .crate_type("rlib")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .arg("-Clinker-plugin-lto")
                    .run();

                let found_files = cwd_filenames();

                let object_files = cwd_object_filenames();
                assert_eq!(object_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(object_files);
                expected_files.insert("libbaz.rlib".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&found_files);
            }

            // unpacked-remapped-split
            // - Debuginfo in `.dwo` files
            // - `.o` and binary refer to remapped `.dwo` paths which do not exist
            // - `.o` deleted
            // - `.dwo` present
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Split,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .split_dwarf_out_dir(split_dwarf_output_directory)
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .run();

                let mut found_files = cwd_filenames();
                found_files.append(&mut dwo_out_filenames(split_dwarf_output_directory));

                let dwo_files = if let Some(dwo_out) = split_dwarf_output_directory {
                    dwo_out_dwo_filenames(dwo_out)
                } else {
                    cwd_dwo_filenames()
                };
                assert_eq!(dwo_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(dwo_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // unpacked-remapped-single
            // - Debuginfo in `.o` files
            // - `.o` and binary refer to remapped `.o` paths which do not exist
            // - `.o` present
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Unspecified,
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .run();

                let found_files = cwd_filenames();

                let object_files = cwd_object_filenames();
                assert_eq!(object_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(object_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // unpacked-remapped-scope
            // - Debuginfo in `.o` files
            // - `.o` and binary refer to remapped `.o` paths which do not exist
            // - `.o` present
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Yes(scope @ "debuginfo"),
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .arg(format!("-Zremap-path-scope={scope}"))
                    .run();

                let found_files = cwd_filenames();

                let object_files = cwd_object_filenames();
                assert_eq!(object_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(object_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::Remapped);
            }

            // unpacked-remapped-wrong-scope
            // - Debuginfo in `.o` files
            // - `.o` and binary refer to un-remapped `.o` paths because remap path scope is macro
            // - `.o` present
            // - `.dwo` never created
            // - `.dwp` never created
            (
                SplitDebuginfo::Unpacked,
                DebuginfoLevel::Full,
                SplitDwarfKind::Single,
                LinkerPluginLto::Unspecified,
                RemapPathPrefix::Yes { remapped_prefix },
                RemapPathScope::Yes(scope @ "macro"),
            ) => {
                rustc(unstable_options)
                    .input("foo.rs")
                    .split_debuginfo(split_kind.cli_value())
                    .debuginfo(level.cli_value())
                    .arg(format!("-Zsplit-dwarf-kind={}", split_dwarf_kind.cli_value()))
                    .remap_path_prefix(cwd(), remapped_prefix)
                    .arg(format!("-Zremap-path-scope={scope}"))
                    .run();

                let found_files = cwd_filenames();

                let object_files = cwd_object_filenames();
                assert_eq!(object_files.len(), 1);

                let mut expected_files = BTreeSet::new();
                expected_files.extend(object_files);
                expected_files.insert("foo".to_string());

                FileAssertions {
                    expected_files: expected_files.iter().map(String::as_ref).collect(),
                }
                .assert_on(&found_files);

                check_path_remap(cwd(), RemapExpectation::NoRemap);
            }

            (split_kind, level, split_dwarf_kind, lto, remap_path_prefix, remap_path_scope) => {
                panic!(
                    "split_kind={:?} + level={:?} + split_dwarf_kind={:?} + lto={:?} + remap_path_prefix={:?} + remap_path_scope={:?} is not handled on linux/other",
                    split_kind, level, split_dwarf_kind, lto, remap_path_prefix, remap_path_scope
                )
            }
        }
    }

    #[derive(PartialEq)]
    enum RemapExpectation {
        Remapped,
        NoRemap,
    }

    #[track_caller]
    fn check_path_remap(cwd_path: PathBuf, remap_expectation: RemapExpectation) {
        let cwd_path = cwd_path.to_str().unwrap();
        let output = llvm_dwarfdump().input("foo").arg("--debug-info").run().stdout_utf8();
        let output_lines: Vec<_> = output.lines().collect();

        // Look for `DW_AT_comp_dir` and `DW_AT_GNU_dwo_name` via `llvm-dwarfdump`. Note: space
        // between uses tabs.
        //
        // ```text
        // 0x0000000b: DW_TAG_compile_unit
        //     DW_AT_stmt_list   (0x00000000)
        //     DW_AT_comp_dir    ("/__MY_REMAPPED_PATH") # this could be e.g. /home/repos/rust/ if not remapped
        //     DW_AT_GNU_dwo_name        ("foo.foo.fc848df41df7a00d-cgu.0.rcgu.dwo")
        // ```
        //
        // FIXME: this is very fragile because the output format can be load-bearing, but doing this
        // via `object` + `gimli` is rather difficult.
        let mut window = output_lines.windows(2);
        while let Some([first_ln, second_ln]) = window.next() {
            let first_ln = first_ln.trim();
            let second_ln = second_ln.trim();

            if !second_ln.starts_with("DW_AT_GNU_dwo_name") {
                continue;
            }

            let Some((comp_dir_attr_name, comp_dir_attr_val)) = first_ln.split_once("\t") else {
                continue;
            };

            println!("comp_dir_attr_name: `{}`", comp_dir_attr_name);
            println!("cwd_path_string: `{}`", cwd_path);

            if comp_dir_attr_name != "DW_AT_comp_dir" {
                continue;
            }

            println!("comp_dir_attr_val: `{}`", comp_dir_attr_val);

            // Possibly `("/__MY_REMAPPED_PATH")` or `($cwd_path_string)`.
            //
            // FIXME: this check is insufficiently precise, it should probably also match on suffix
            // (`.o` vs `.dwo` reference). But also, string matching is just very fragile.
            let comp_dir_attr_val = comp_dir_attr_val.trim();

            match remap_expectation {
                RemapExpectation::Remapped => {
                    assert!(
                        !comp_dir_attr_val.contains(&cwd_path),
                        "unexpected non-remapped path found in `DW_AT_comp_dir`: {}",
                        comp_dir_attr_val
                    );
                }
                RemapExpectation::NoRemap => {
                    assert!(
                        comp_dir_attr_val.contains(&cwd_path),
                        "failed to find un-remapped path in `DW_AT_comp_dir`: {}",
                        comp_dir_attr_val
                    );
                }
            }
        }
    }
}

fn main() {
    // ENHANCEMENT: we are only checking that split-debuginfo is splitting is some way, shape or
    // form, we do not sanity check the actual debuginfo artifacts on non-Linux! It may be possible
    // to also sanity check the debuginfo artifacts, but I did not want to do that during the port
    // to rmake.rs initially.

    // ENHANCEMENT: the Linux checks have significantly more coverage for interaction between `-C
    // split-debuginfo` and other flags compared to windows-msvc or windows-gnu or darwin or some
    // other non-linux targets. It would be cool if their test coverage could be improved.

    // NOTE: these combinations are not exhaustive, because while porting to rmake.rs initially I
    // tried to preserve the existing test behavior closely. Notably, no attempt was made to
    // exhaustively cover all cases in the 6-fold Cartesian product of `{,-Csplit=debuginfo=...}` x
    // `{,-Cdebuginfo=...}` x `{,--remap-path-prefix}` x `{,-Zremap-path-scope=...}` x
    // `{,-Zsplit-dwarf-kind=...}` x `{,-Clinker-plugin-lto}`. If you really want to, you can
    // identify which combination isn't exercised with a 6-layers nested for loop iterating through
    // each of the cli flag enum variants.

    if is_windows_msvc() {
        // FIXME: the windows-msvc test coverage is sparse at best.

        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::Unspecified);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::Unspecified);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::Unspecified);

        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::None);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::None);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::None);

        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::Full);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::Full);
        windows_msvc_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::Full);
    } else if is_windows() {
        // FIXME(#135531): the `Makefile` version didn't test windows at all. I don't know about the
        // intended behavior on windows-gnu to expand test coverage while porting this to rmake.rs,
        // but the test coverage here really should be expanded since some windows-gnu targets are
        // Tier 1.
    } else if is_darwin() {
        // FIXME: the darwin test coverage is sparse at best.

        // Expect no `.dSYM` generation if debuginfo is not requested (special case).
        darwin_tests::split_debuginfo(SplitDebuginfo::Unspecified, DebuginfoLevel::Unspecified);

        darwin_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::Unspecified);
        darwin_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::Unspecified);
        darwin_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::Unspecified);

        darwin_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::None);
        darwin_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::None);
        darwin_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::None);

        darwin_tests::split_debuginfo(SplitDebuginfo::Off, DebuginfoLevel::Full);
        darwin_tests::split_debuginfo(SplitDebuginfo::Unpacked, DebuginfoLevel::Full);
        darwin_tests::split_debuginfo(SplitDebuginfo::Packed, DebuginfoLevel::Full);
    } else {
        // Unix as well as the non-linux + non-windows + non-darwin targets.

        // FIXME: this `uname` check is very funny, it really should be refined (e.g. llvm bug
        // <https://github.com/llvm/llvm-project/issues/56642> for riscv64 targets).

        // NOTE: some options are not stable on non-linux + non-windows + non-darwin targets...
        let unstable_options =
            if uname() == "Linux" { UnstableOptions::Unspecified } else { UnstableOptions::Yes };

        // FIXME: we should add a test with scope `split-debuginfo,split-debuginfo-path` that greps
        // the entire `.dwp` file for remapped paths (i.e. without going through objdump or
        // readelf). See <https://github.com/rust-lang/rust/pull/118518#discussion_r1452180392>.

        use shared_linux_other_tests::CrossCrateTest;

        // unspecified `-Csplit-debuginfo`
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unspecified,
            DebuginfoLevel::Full,
            SplitDwarfKind::Unspecified,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // off
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Off,
            DebuginfoLevel::Full,
            SplitDwarfKind::Unspecified,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // packed-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // packed-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // packed-lto-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Yes,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // packed-lto-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Yes,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // FIXME: the remapping tests probably need to be reworked, see
        // <https://github.com/rust-lang/rust/pull/118518#discussion_r1452174338>.

        // packed-remapped-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Unspecified,
            None,
        );

        // packed-remapped-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Unspecified,
            None,
        );

        // packed-remapped-scope
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Yes("debuginfo"),
            None,
        );

        // packed-remapped-wrong-scope
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Yes("macro"),
            None,
        );

        // packed-crosscrate-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::Yes,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // packed-crosscrate-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::Yes,
            unstable_options,
            SplitDebuginfo::Packed,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-split with split-dwarf-out-dir
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            UnstableOptions::Yes,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            Some("other-dir"),
        );

        // unpacked-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-lto-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Yes,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-lto-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Yes,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-remapped-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-remapped-split with split-dwarf-out-dir
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            UnstableOptions::Yes,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Unspecified,
            Some("other-dir"),
        );

        // unpacked-remapped-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-remapped-scope
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Yes("debuginfo"),
            None,
        );

        // unpacked-remapped-wrong-scope
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::No,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Yes { remapped_prefix: "/__MY_REMAPPED_PATH__" },
            RemapPathScope::Yes("macro"),
            None,
        );

        // unpacked-crosscrate-split
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::Yes,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );

        // unpacked-crosscrate-split with split-dwarf-out-dir
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::Yes,
            UnstableOptions::Yes,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Split,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            Some("other-dir"),
        );

        // unpacked-crosscrate-single
        shared_linux_other_tests::split_debuginfo(
            CrossCrateTest::Yes,
            unstable_options,
            SplitDebuginfo::Unpacked,
            DebuginfoLevel::Full,
            SplitDwarfKind::Single,
            LinkerPluginLto::Unspecified,
            RemapPathPrefix::Unspecified,
            RemapPathScope::Unspecified,
            None,
        );
    }
}
