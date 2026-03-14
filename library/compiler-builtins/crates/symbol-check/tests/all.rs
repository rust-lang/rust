use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use assert_cmd::assert::Assert;
use assert_cmd::cargo::cargo_bin_cmd;
use object::BinaryFormat;
use tempfile::tempdir;

trait AssertExt {
    fn stderr_contains(self, s: &str) -> Self;
}

impl AssertExt for Assert {
    #[track_caller]
    fn stderr_contains(self, s: &str) -> Self {
        let out = String::from_utf8_lossy(&self.get_output().stderr);
        assert!(out.contains(s), "looking for: `{s}`\nout:\n```\n{out}\n```");
        self
    }
}

#[test]
fn test_duplicates() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let dup_out = dir.path().join("dup.o");
    let lib_out = dir.path().join("libfoo.rlib");

    // For the "bad" file, we need duplicate symbols from different object files in the archive. Do
    // this reliably by building an archive and a separate object file then merging them.
    t.rustc_build(&input_dir().join("duplicates.rs"), &lib_out, |cmd| cmd);
    t.rustc_build(&input_dir().join("duplicates.rs"), &dup_out, |cmd| {
        cmd.arg("--emit=obj")
    });

    let mut ar = t.cc_build().get_archiver();

    if ar.get_program().to_string_lossy().contains("lib.exe") {
        let mut out_arg = OsString::from("-out:");
        out_arg.push(&lib_out);
        ar.arg(&out_arg);
        // Repeating the same file as the first arg makes lib.exe append (taken from the
        // `cc` implementation).
        ar.arg(&lib_out);
    } else {
        ar.arg("rs")
            // Eat an `libfoo.rlib(lib.rmeta) has no symbols` info message on MacOS
            .stderr(Stdio::null())
            .arg(&lib_out);
    }

    run(ar.arg(&dup_out));

    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("duplicate symbols")
        .stderr_contains("FDUP")
        .stderr_contains("IDUP")
        .stderr_contains("fndup");
}

#[test]
fn test_core_symbols() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    t.rustc_build(&input_dir().join("core_symbols.rs"), &lib_out, |cmd| cmd);
    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("found 1 undefined symbols from core")
        .stderr_contains("from_utf8");
}

#[test]
fn test_visible_symbols() {
    let t = TestTarget::from_env();
    if t.is_windows() {
        eprintln!("windows does not have visibility, skipping");
        return;
    }
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    t.rustc_build(&input_dir().join("good_lib.rs"), &lib_out, |cmd| cmd);
    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert.failure().stderr_contains("found 1 visible symbols"); // good is visible.
}

mod exe_stack {
    use super::*;

    /// Check with an object that has no `.note.GNU-stack` section, indicating platform-default stack
    /// writeability (usually enabled).
    #[test]
    fn test_missing_gnu_stack_section() {
        let t = TestTarget::from_env();
        if t.is_msvc() {
            // Can't easily build asm via cc with cl.exe / masm.exe
            eprintln!("assembly on windows, skipping");
            return;
        }

        let dir = tempdir().unwrap();
        let src = input_dir().join("missing_gnu_stack_section.S");

        let objs = t.cc_build().file(src).out_dir(&dir).compile_intermediates();
        let [obj] = objs.as_slice() else { panic!() };

        let assert = t.symcheck_exe().arg(obj).arg("--no-visibility").assert();

        if t.is_ppc64be() || t.no_os() || t.binary_obj_format() != BinaryFormat::Elf {
            // Ppc64be doesn't emit `.note.GNU-stack`, not relevant without an OS, and non-elf
            // targets don't use `.note.GNU-stack`.
            assert.success();
            return;
        }

        assert
            .failure()
            .stderr_contains("the following object files require an executable stack")
            .stderr_contains("missing_gnu_stack_section.o (no .note.GNU-stack section)");
    }

    /// Check with an object that has a `.note.GNU-stack` section with the executable flag set.
    #[test]
    fn test_exe_gnu_stack_section() {
        let t = TestTarget::from_env();
        let mut build = t.cc_build();
        if !build.get_compiler().is_like_gnu() || t.is_windows() {
            eprintln!("unsupported compiler for nested functions, skipping");
            return;
        }

        let dir = tempdir().unwrap();
        let objs = build
            .file(input_dir().join("has_exe_gnu_stack_section.c"))
            .out_dir(&dir)
            .compile_intermediates();
        let [obj] = objs.as_slice() else { panic!() };

        let assert = t.symcheck_exe().arg(obj).arg("--no-visibility").assert();

        if t.is_ppc64be() || t.no_os() {
            // Ppc64be doesn't emit `.note.GNU-stack`, not relevant without an OS.
            assert.success();
            return;
        }

        assert
            .failure()
            .stderr_contains("the following object files require an executable stack")
            .stderr_contains(
                "has_exe_gnu_stack_section.o (.note.GNU-stack section marked SHF_EXECINSTR)",
            );
    }

    /// Check a final binary with `PT_GNU_STACK`.
    #[test]
    fn test_execstack_bin() {
        let t = TestTarget::from_env();
        if t.binary_obj_format() != BinaryFormat::Elf || !t.supports_executables() {
            // Mac's Clang rejects `-z execstack`. `-allow_stack_execute` should work per the ld
            // manpage, at least on x86, but it doesn't seem to., not relevant without an OS.
            eprintln!("non-elf or no-executable target, skipping");
            return;
        }

        let dir = tempdir().unwrap();
        let out = dir.path().join("execstack.out");

        let mut cmd = t.cc_build().get_compiler().to_command();
        t.set_bin_out_path(&mut cmd, &out);

        run(cmd
            .arg("-z")
            .arg("execstack")
            .arg(input_dir().join("good_bin.c")));

        let assert = t.symcheck_exe().arg(&out).assert();
        assert
            .failure()
            .stderr_contains("the following object files require an executable stack")
            .stderr_contains("execstack.out (PT_GNU_STACK program header marked PF_X)");
    }
}

#[test]
fn test_good_lib() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    t.rustc_build(&input_dir().join("good_lib.rs"), &lib_out, |cmd| cmd);
    let assert = t
        .symcheck_exe()
        .arg(&lib_out)
        .arg("--no-visibility")
        .assert();
    assert.success();
}

#[test]
fn test_good_bin() {
    let t = TestTarget::from_env();
    // Nothing to test if we can't build a binary.
    if !t.supports_executables() {
        eprintln!("no-exe target, skipping");
        return;
    }

    let dir = tempdir().unwrap();
    let out = dir.path().join("good_bin.out");

    let mut cmd = t.cc_build().get_compiler().to_command();
    t.set_bin_out_path(&mut cmd, &out);
    run(cmd.arg(input_dir().join("good_bin.c")));

    let assert = t.symcheck_exe().arg(&out).arg("--no-visibility").assert();
    assert.success();
}

/// Since symcheck is a hostprog, the target we want to build and test symcheck for may not be the
/// same as the host target.
struct TestTarget {
    triple: String,
}

impl TestTarget {
    fn from_env() -> Self {
        let triple = match env::var("SYMCHECK_TEST_TARGET") {
            Ok(t) => t,
            // Require on CI so we don't accidentally always test the native target
            _ if env::var("CI").is_ok() => panic!("SYMCHECK_TEST_TARGET must be set in CI"),
            // Fall back to native for local convenience.
            Err(_) => env!("HOST").to_string(),
        };

        println!("using target {triple}");
        Self { triple }
    }

    /// Build i -> o with optional additional configuration.
    fn rustc_build(&self, i: &Path, o: &Path, mut f: impl FnMut(&mut Command) -> &mut Command) {
        let mut cmd = Command::new("rustc");
        cmd.arg(i)
            .arg("--target")
            .arg(&self.triple)
            .arg("--crate-type=lib")
            .arg("-o")
            .arg(o);
        f(&mut cmd);
        run(&mut cmd);
    }

    /// Configure `cc` with the host and target.
    fn cc_build(&self) -> cc::Build {
        let mut b = cc::Build::new();
        b.host(env!("HOST"))
            .target(&self.triple)
            .opt_level(0)
            .cargo_debug(true)
            .cargo_metadata(false);
        b
    }

    fn symcheck_exe(&self) -> assert_cmd::Command {
        let mut cmd = cargo_bin_cmd!();
        cmd.arg("--check");
        if self.no_os() {
            cmd.arg("--no-os");
        }
        cmd
    }

    /// MSVC requires different flags for setting output path, account for that here.
    fn set_bin_out_path<'a>(&self, cmd: &'a mut Command, out: &Path) -> &'a mut Command {
        if self.cc_build().get_compiler().is_like_msvc() {
            let mut exe_arg = OsString::from("/Fe");
            let mut obj_arg = OsString::from("/Fo");
            exe_arg.push(out);
            obj_arg.push(out.with_extension("o"));
            cmd.arg(exe_arg).arg(obj_arg)
        } else {
            cmd.arg("-o").arg(out)
        }
    }

    /// Based on `rustc_target`.
    fn binary_obj_format(&self) -> BinaryFormat {
        let t = &self.triple;
        if t.contains("-windows-") || t.contains("-cygwin") {
            // Coff for libraries, PE for executables.
            BinaryFormat::Coff
        } else if t.starts_with("wasm") {
            BinaryFormat::Wasm
        } else if t.contains("-aix") {
            BinaryFormat::Xcoff
        } else if t.contains("-apple-") {
            BinaryFormat::MachO
        } else {
            BinaryFormat::Elf
        }
    }

    fn is_windows(&self) -> bool {
        self.triple.contains("-windows-")
    }

    fn is_msvc(&self) -> bool {
        self.triple.contains("-windows-msvc")
    }

    fn is_ppc64be(&self) -> bool {
        self.triple.starts_with("powerpc64-")
    }

    /// True if the target needs `--no-os` passed to symcheck.
    fn no_os(&self) -> bool {
        self.triple.contains("-none")
    }

    /// True if the target supports (easily) building to a final executable.
    fn supports_executables(&self) -> bool {
        // Technically i686-pc-windows-gnu should work but it has nontrivial setup in CI.
        !(self.no_os()
            || self.triple == "wasm32-unknown-unknown"
            || self.triple == "i686-pc-windows-gnu")
    }
}

fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/input")
}

#[track_caller]
fn run(cmd: &mut Command) {
    eprintln!("+ {cmd:?}");
    let out = cmd.output().unwrap();
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{:?}", out.status);
}
