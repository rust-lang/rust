//! `run-make-support` is a support library for run-make tests. It provides command wrappers and
//! convenience utility functions to help test writers reduce duplication. The support library
//! notably is built via cargo: this means that if your test wants some non-trivial utility, such
//! as `object` or `wasmparser`, they can be re-exported and be made available through this library.

pub mod cc;
pub mod clang;
mod command;
pub mod diff;
mod drop_bomb;
pub mod fs_wrapper;
pub mod llvm;
pub mod run;
pub mod rustc;
pub mod rustdoc;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::panic;
use std::path::{Path, PathBuf};

pub use gimli;
pub use object;
pub use regex;
pub use wasmparser;

pub use cc::{cc, extra_c_flags, extra_cxx_flags, Cc};
pub use clang::{clang, Clang};
pub use diff::{diff, Diff};
pub use llvm::{
    llvm_filecheck, llvm_objdump, llvm_profdata, llvm_readobj, LlvmFilecheck, LlvmObjdump,
    LlvmProfdata, LlvmReadobj,
};
pub use run::{cmd, run, run_fail, run_with_args};
pub use rustc::{aux_build, rustc, Rustc};
pub use rustdoc::{bare_rustdoc, rustdoc, Rustdoc};

#[track_caller]
#[must_use]
pub fn env_var(name: &str) -> String {
    match env::var(name) {
        Ok(v) => v,
        Err(err) => panic!("failed to retrieve environment variable {name:?}: {err:?}"),
    }
}

#[track_caller]
#[must_use]
pub fn env_var_os(name: &str) -> OsString {
    match env::var_os(name) {
        Some(v) => v,
        None => panic!("failed to retrieve environment variable {name:?}"),
    }
}

/// `TARGET`
#[must_use]
pub fn target() -> String {
    env_var("TARGET")
}

/// `AR`
#[track_caller]
pub fn ar(inputs: &[impl AsRef<Path>], output_path: impl AsRef<Path>) {
    let output = fs::File::create(&output_path).expect(&format!(
        "the file in path \"{}\" could not be created",
        output_path.as_ref().display()
    ));
    let mut builder = ar::Builder::new(output);
    for input in inputs {
        builder.append_path(input).unwrap();
    }
}

/// Check if target is windows-like.
#[must_use]
pub fn is_windows() -> bool {
    target().contains("windows")
}

/// Check if target uses msvc.
#[must_use]
pub fn is_msvc() -> bool {
    target().contains("msvc")
}

/// Check if target uses macOS.
#[must_use]
pub fn is_darwin() -> bool {
    target().contains("darwin")
}

#[track_caller]
#[must_use]
pub fn python_command() -> Command {
    let python_path = env_var("PYTHON");
    Command::new(python_path)
}

#[track_caller]
#[must_use]
pub fn htmldocck() -> Command {
    let mut python = python_command();
    python.arg(source_root().join("src/etc/htmldocck.py"));
    python
}

/// Returns the path for a local test file.
pub fn path<P: AsRef<Path>>(p: P) -> PathBuf {
    cwd().join(p.as_ref())
}

/// Path to the root rust-lang/rust source checkout.
#[must_use]
pub fn source_root() -> PathBuf {
    env_var("SOURCE_ROOT").into()
}

/// Creates a new symlink to a path on the filesystem, adjusting for Windows or Unix.
#[cfg(target_family = "windows")]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    if link.as_ref().exists() {
        std::fs::remove_dir(link.as_ref()).unwrap();
    }
    use std::os::windows::fs;
    fs::symlink_file(original.as_ref(), link.as_ref()).expect(&format!(
        "failed to create symlink {:?} for {:?}",
        link.as_ref().display(),
        original.as_ref().display(),
    ));
}

/// Creates a new symlink to a path on the filesystem, adjusting for Windows or Unix.
#[cfg(target_family = "unix")]
pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) {
    if link.as_ref().exists() {
        std::fs::remove_dir(link.as_ref()).unwrap();
    }
    use std::os::unix::fs;
    fs::symlink(original.as_ref(), link.as_ref()).expect(&format!(
        "failed to create symlink {:?} for {:?}",
        link.as_ref().display(),
        original.as_ref().display(),
    ));
}

/// Construct the static library name based on the platform.
#[must_use]
pub fn static_lib_name(name: &str) -> String {
    // See tools.mk (irrelevant lines omitted):
    //
    // ```makefile
    // ifeq ($(UNAME),Darwin)
    //     STATICLIB = $(TMPDIR)/lib$(1).a
    // else
    //     ifdef IS_WINDOWS
    //         ifdef IS_MSVC
    //             STATICLIB = $(TMPDIR)/$(1).lib
    //         else
    //             STATICLIB = $(TMPDIR)/lib$(1).a
    //         endif
    //     else
    //         STATICLIB = $(TMPDIR)/lib$(1).a
    //     endif
    // endif
    // ```
    assert!(!name.contains(char::is_whitespace), "static library name cannot contain whitespace");

    if is_msvc() { format!("{name}.lib") } else { format!("lib{name}.a") }
}

/// Construct the dynamic library name based on the platform.
#[must_use]
pub fn dynamic_lib_name(name: &str) -> String {
    // See tools.mk (irrelevant lines omitted):
    //
    // ```makefile
    // ifeq ($(UNAME),Darwin)
    //     DYLIB = $(TMPDIR)/lib$(1).dylib
    // else
    //     ifdef IS_WINDOWS
    //         DYLIB = $(TMPDIR)/$(1).dll
    //     else
    //         DYLIB = $(TMPDIR)/lib$(1).so
    //     endif
    // endif
    // ```
    assert!(!name.contains(char::is_whitespace), "dynamic library name cannot contain whitespace");

    let extension = dynamic_lib_extension();
    if is_darwin() {
        format!("lib{name}.{extension}")
    } else if is_windows() {
        format!("{name}.{extension}")
    } else {
        format!("lib{name}.{extension}")
    }
}

#[must_use]
pub fn dynamic_lib_extension() -> &'static str {
    if is_darwin() {
        "dylib"
    } else if is_windows() {
        "dll"
    } else {
        "so"
    }
}

/// Generate the name a rust library (rlib) would have.
#[must_use]
pub fn rust_lib_name(name: &str) -> String {
    format!("lib{name}.rlib")
}

/// Construct the binary name based on platform.
#[must_use]
pub fn bin_name(name: &str) -> String {
    if is_windows() { format!("{name}.exe") } else { name.to_string() }
}

/// Return the current working directory.
#[must_use]
pub fn cwd() -> PathBuf {
    env::current_dir().unwrap()
}

// FIXME(Oneirical): This will no longer be required after compiletest receives the ability
// to manipulate read-only files. See https://github.com/rust-lang/rust/issues/126334
/// Ensure that the path P is read-only while the test runs, and restore original permissions
/// at the end so compiletest can clean up.
/// This will panic on Windows if the path is a directory (as it would otherwise do nothing)
#[track_caller]
pub fn test_while_readonly<P: AsRef<Path>, F: FnOnce() + std::panic::UnwindSafe>(
    path: P,
    closure: F,
) {
    let path = path.as_ref();
    if is_windows() && path.is_dir() {
        eprintln!("This helper function cannot be used on Windows to make directories readonly.");
        eprintln!(
            "See the official documentation:
            https://doc.rust-lang.org/std/fs/struct.Permissions.html#method.set_readonly"
        );
        panic!("`test_while_readonly` on directory detected while on Windows.");
    }
    let metadata = fs_wrapper::metadata(&path);
    let original_perms = metadata.permissions();

    let mut new_perms = original_perms.clone();
    new_perms.set_readonly(true);
    fs_wrapper::set_permissions(&path, new_perms);

    let success = std::panic::catch_unwind(closure);

    fs_wrapper::set_permissions(&path, original_perms);
    success.unwrap();
}

/// Use `cygpath -w` on a path to get a Windows path string back. This assumes that `cygpath` is
/// available on the platform!
#[track_caller]
#[must_use]
pub fn cygpath_windows<P: AsRef<Path>>(path: P) -> String {
    let caller = panic::Location::caller();
    let mut cygpath = Command::new("cygpath");
    cygpath.arg("-w");
    cygpath.arg(path.as_ref());
    let output = cygpath.run();
    if !output.status().success() {
        handle_failed_output(&cygpath, output, caller.line());
    }
    // cygpath -w can attach a newline
    output.stdout_utf8().trim().to_string()
}

/// Run `uname`. This assumes that `uname` is available on the platform!
#[track_caller]
#[must_use]
pub fn uname() -> String {
    let caller = panic::Location::caller();
    let mut uname = Command::new("uname");
    let output = uname.run();
    if !output.status().success() {
        handle_failed_output(&uname, output, caller.line());
    }
    output.stdout_utf8()
}

fn handle_failed_output(cmd: &Command, output: CompletedProcess, caller_line_number: u32) -> ! {
    if output.status().success() {
        eprintln!("command unexpectedly succeeded at line {caller_line_number}");
    } else {
        eprintln!("command failed at line {caller_line_number}");
    }
    eprintln!("{cmd:?}");
    eprintln!("output status: `{}`", output.status());
    eprintln!("=== STDOUT ===\n{}\n\n", output.stdout_utf8());
    eprintln!("=== STDERR ===\n{}\n\n", output.stderr_utf8());
    std::process::exit(1)
}

/// Set the runtime library path as needed for running the host rustc/rustdoc/etc.
pub fn set_host_rpath(cmd: &mut Command) {
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(cwd());
        paths.push(PathBuf::from(env_var("HOST_RPATH_DIR")));
        for p in env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });
}

/// Read the contents of a file that cannot simply be read by
/// read_to_string, due to invalid utf8 data, then assert that it contains `expected`.
#[track_caller]
pub fn invalid_utf8_contains<P: AsRef<Path>>(path: P, expected: &str) {
    let buffer = fs_wrapper::read(path.as_ref());
    if !String::from_utf8_lossy(&buffer).contains(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&buffer));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was not found in file");
    }
}

/// Read the contents of a file that cannot simply be read by
/// read_to_string, due to invalid utf8 data, then assert that it does not contain `expected`.
#[track_caller]
pub fn invalid_utf8_not_contains<P: AsRef<Path>>(path: P, expected: &str) {
    let buffer = fs_wrapper::read(path.as_ref());
    if String::from_utf8_lossy(&buffer).contains(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&buffer));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was unexpectedly found in file");
    }
}

/// Copy a directory into another.
pub fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    fn copy_dir_all_inner(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
        let dst = dst.as_ref();
        if !dst.is_dir() {
            std::fs::create_dir_all(&dst)?;
        }
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_dir() {
                copy_dir_all_inner(entry.path(), dst.join(entry.file_name()))?;
            } else {
                std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
            }
        }
        Ok(())
    }

    if let Err(e) = copy_dir_all_inner(&src, &dst) {
        // Trying to give more context about what exactly caused the failure
        panic!(
            "failed to copy `{}` to `{}`: {:?}",
            src.as_ref().display(),
            dst.as_ref().display(),
            e
        );
    }
}

/// Check that all files in `dir1` exist and have the same content in `dir2`. Panic otherwise.
pub fn recursive_diff(dir1: impl AsRef<Path>, dir2: impl AsRef<Path>) {
    let dir2 = dir2.as_ref();
    read_dir(dir1, |entry_path| {
        let entry_name = entry_path.file_name().unwrap();
        if entry_path.is_dir() {
            recursive_diff(&entry_path, &dir2.join(entry_name));
        } else {
            let path2 = dir2.join(entry_name);
            let file1 = fs_wrapper::read(&entry_path);
            let file2 = fs_wrapper::read(&path2);

            // We don't use `assert_eq!` because they are `Vec<u8>`, so not great for display.
            // Why not using String? Because there might be minified files or even potentially
            // binary ones, so that would display useless output.
            assert!(
                file1 == file2,
                "`{}` and `{}` have different content",
                entry_path.display(),
                path2.display(),
            );
        }
    });
}

pub fn read_dir<F: FnMut(&Path)>(dir: impl AsRef<Path>, mut callback: F) {
    for entry in fs_wrapper::read_dir(dir) {
        callback(&entry.unwrap().path());
    }
}

/// Check that `actual` is equal to `expected`. Panic otherwise.
#[track_caller]
pub fn assert_equals(actual: &str, expected: &str) {
    if actual != expected {
        eprintln!("=== ACTUAL TEXT ===");
        eprintln!("{}", actual);
        eprintln!("=== EXPECTED ===");
        eprintln!("{}", expected);
        panic!("expected text was not found in actual text");
    }
}

/// Check that `haystack` contains `needle`. Panic otherwise.
#[track_caller]
pub fn assert_contains(haystack: &str, needle: &str) {
    if !haystack.contains(needle) {
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", needle);
        panic!("needle was not found in haystack");
    }
}

/// Check that `haystack` does not contain `needle`. Panic otherwise.
#[track_caller]
pub fn assert_not_contains(haystack: &str, needle: &str) {
    if haystack.contains(needle) {
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", needle);
        panic!("needle was unexpectedly found in haystack");
    }
}

/// This function is designed for running commands in a temporary directory
/// that is cleared after the function ends.
///
/// What this function does:
/// 1) Creates a temporary directory (`tmpdir`)
/// 2) Copies all files from the current directory to `tmpdir`
/// 3) Changes the current working directory to `tmpdir`
/// 4) Calls `callback`
/// 5) Switches working directory back to the original one
/// 6) Removes `tmpdir`
pub fn run_in_tmpdir<F: FnOnce()>(callback: F) {
    let original_dir = cwd();
    let tmpdir = original_dir.join("../temporary-directory");
    copy_dir_all(".", &tmpdir);

    env::set_current_dir(&tmpdir).unwrap();
    callback();
    env::set_current_dir(original_dir).unwrap();
    fs::remove_dir_all(tmpdir).unwrap();
}

/// Implement common helpers for command wrappers. This assumes that the command wrapper is a struct
/// containing a `cmd: Command` field. The provided helpers are:
///
/// 1. Generic argument acceptors: `arg` and `args` (delegated to [`Command`]). These are intended
///    to be *fallback* argument acceptors, when specific helpers don't make sense. Prefer to add
///    new specific helper methods over relying on these generic argument providers.
/// 2. Environment manipulation methods: `env`, `env_remove` and `env_clear`: these delegate to
///    methods of the same name on [`Command`].
/// 3. Output and execution: `run` and `run_fail` are provided. These are
///    higher-level convenience methods which wait for the command to finish running and assert
///    that the command successfully ran or failed as expected. They return
///    [`CompletedProcess`], which can be used to assert the stdout/stderr/exit code of the executed
///    process.
///
/// Example usage:
///
/// ```ignore (illustrative)
/// struct CommandWrapper { cmd: Command } // <- required `cmd` field
///
/// crate::impl_common_helpers!(CommandWrapper);
///
/// impl CommandWrapper {
///     // ... additional specific helper methods
/// }
/// ```
macro_rules! impl_common_helpers {
    ($wrapper: ident) => {
        impl $wrapper {
            /// Specify an environment variable.
            pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
            where
                K: AsRef<::std::ffi::OsStr>,
                V: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.env(key, value);
                self
            }

            /// Remove an environmental variable.
            pub fn env_remove<K>(&mut self, key: K) -> &mut Self
            where
                K: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.env_remove(key);
                self
            }

            /// Generic command argument provider. Prefer specific helper methods if possible.
            /// Note that for some executables, arguments might be platform specific. For C/C++
            /// compilers, arguments might be platform *and* compiler specific.
            pub fn arg<S>(&mut self, arg: S) -> &mut Self
            where
                S: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.arg(arg);
                self
            }

            /// Generic command arguments provider. Prefer specific helper methods if possible.
            /// Note that for some executables, arguments might be platform specific. For C/C++
            /// compilers, arguments might be platform *and* compiler specific.
            pub fn args<S>(&mut self, args: &[S]) -> &mut Self
            where
                S: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.args(args);
                self
            }

            /// Inspect what the underlying [`Command`] is up to the
            /// current construction.
            pub fn inspect<I>(&mut self, inspector: I) -> &mut Self
            where
                I: FnOnce(&::std::process::Command),
            {
                self.cmd.inspect(inspector);
                self
            }

            /// Run the constructed command and assert that it is successfully run.
            #[track_caller]
            pub fn run(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run()
            }

            /// Run the constructed command and assert that it does not successfully run.
            #[track_caller]
            pub fn run_fail(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run_fail()
            }

            /// Run the command but do not check its exit status.
            /// Only use if you explicitly don't care about the exit status.
            /// Prefer to use [`Self::run`] and [`Self::run_fail`]
            /// whenever possible.
            #[track_caller]
            pub fn run_unchecked(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run_unchecked()
            }

            /// Set the path where the command will be run.
            pub fn current_dir<P: AsRef<::std::path::Path>>(&mut self, path: P) -> &mut Self {
                self.cmd.current_dir(path);
                self
            }
        }
    };
}

use crate::command::{Command, CompletedProcess};
pub(crate) use impl_common_helpers;
