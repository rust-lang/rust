//! `run-make-support` is a support library for run-make tests. It provides command wrappers and
//! convenience utility functions to help test writers reduce duplication. The support library
//! notably is built via cargo: this means that if your test wants some non-trivial utility, such
//! as `object` or `wasmparser`, they can be re-exported and be made available through this library.

pub mod cc;
pub mod clang;
pub mod diff;
pub mod llvm_readobj;
pub mod run;
pub mod rustc;
pub mod rustdoc;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub use gimli;
pub use object;
pub use regex;
pub use wasmparser;

pub use cc::{cc, extra_c_flags, extra_cxx_flags, Cc};
pub use clang::{clang, Clang};
pub use diff::{diff, Diff};
pub use llvm_readobj::{llvm_readobj, LlvmReadobj};
pub use run::{run, run_fail};
pub use rustc::{aux_build, rustc, Rustc};
pub use rustdoc::{bare_rustdoc, rustdoc, Rustdoc};

pub fn env_var(name: &str) -> String {
    match env::var(name) {
        Ok(v) => v,
        Err(err) => panic!("failed to retrieve environment variable {name:?}: {err:?}"),
    }
}

pub fn env_var_os(name: &str) -> OsString {
    match env::var_os(name) {
        Some(v) => v,
        None => panic!("failed to retrieve environment variable {name:?}"),
    }
}

/// Path of `TMPDIR` (a temporary build directory, not under `/tmp`).
pub fn tmp_dir() -> PathBuf {
    env_var_os("TMPDIR").into()
}

/// `TARGET`
pub fn target() -> String {
    env_var("TARGET")
}

/// Check if target is windows-like.
pub fn is_windows() -> bool {
    target().contains("windows")
}

/// Check if target uses msvc.
pub fn is_msvc() -> bool {
    target().contains("msvc")
}

/// Check if target uses macOS.
pub fn is_darwin() -> bool {
    target().contains("darwin")
}

/// Construct a path to a static library under `$TMPDIR` given the library name. This will return a
/// path with `$TMPDIR` joined with platform-and-compiler-specific library name.
pub fn static_lib(name: &str) -> PathBuf {
    tmp_dir().join(static_lib_name(name))
}

pub fn python_command() -> Command {
    let python_path = env_var("PYTHON");
    Command::new(python_path)
}

pub fn htmldocck() -> Command {
    let mut python = python_command();
    python.arg(source_path().join("src/etc/htmldocck.py"));
    python
}

pub fn source_path() -> PathBuf {
    env_var("S").into()
}

/// Construct the static library name based on the platform.
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

/// Construct a path to a dynamic library under `$TMPDIR` given the library name. This will return a
/// path with `$TMPDIR` joined with platform-and-compiler-specific library name.
pub fn dynamic_lib(name: &str) -> PathBuf {
    tmp_dir().join(dynamic_lib_name(name))
}

/// Construct the dynamic library name based on the platform.
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

pub fn dynamic_lib_extension() -> &'static str {
    if is_darwin() {
        "dylib"
    } else if is_windows() {
        "dll"
    } else {
        "so"
    }
}

/// Construct a path to a rust library (rlib) under `$TMPDIR` given the library name. This will return a
/// path with `$TMPDIR` joined with the library name.
pub fn rust_lib(name: &str) -> PathBuf {
    tmp_dir().join(rust_lib_name(name))
}

/// Generate the name a rust library (rlib) would have. If you want the complete path, use
/// [`rust_lib`] instead.
pub fn rust_lib_name(name: &str) -> String {
    format!("lib{name}.rlib")
}

/// Construct the binary name based on platform.
pub fn bin_name(name: &str) -> String {
    if is_windows() { format!("{name}.exe") } else { name.to_string() }
}

/// Use `cygpath -w` on a path to get a Windows path string back. This assumes that `cygpath` is
/// available on the platform!
#[track_caller]
pub fn cygpath_windows<P: AsRef<Path>>(path: P) -> String {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let mut cygpath = Command::new("cygpath");
    cygpath.arg("-w");
    cygpath.arg(path.as_ref());
    let output = cygpath.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&cygpath, output, caller_line_number);
    }
    let s = String::from_utf8(output.stdout).unwrap();
    // cygpath -w can attach a newline
    s.trim().to_string()
}

/// Run `uname`. This assumes that `uname` is available on the platform!
#[track_caller]
pub fn uname() -> String {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let mut uname = Command::new("uname");
    let output = uname.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&uname, output, caller_line_number);
    }
    String::from_utf8(output.stdout).unwrap()
}

fn handle_failed_output(cmd: &Command, output: Output, caller_line_number: u32) -> ! {
    if output.status.success() {
        eprintln!("command unexpectedly succeeded at line {caller_line_number}");
    } else {
        eprintln!("command failed at line {caller_line_number}");
    }
    eprintln!("{cmd:?}");
    eprintln!("output status: `{}`", output.status);
    eprintln!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    eprintln!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}

/// Set the runtime library path as needed for running the host rustc/rustdoc/etc.
pub fn set_host_rpath(cmd: &mut Command) {
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(PathBuf::from(env_var("TMPDIR")));
        paths.push(PathBuf::from(env_var("HOST_RPATH_DIR")));
        for p in env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });
}

/// Copy a directory into another.
pub fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    fn copy_dir_all_inner(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<()> {
        let dst = dst.as_ref();
        if !dst.is_dir() {
            fs::create_dir_all(&dst)?;
        }
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_dir() {
                copy_dir_all_inner(entry.path(), dst.join(entry.file_name()))?;
            } else {
                fs::copy(entry.path(), dst.join(entry.file_name()))?;
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
    fn read_file(path: &Path) -> Vec<u8> {
        match fs::read(path) {
            Ok(c) => c,
            Err(e) => panic!("Failed to read `{}`: {:?}", path.display(), e),
        }
    }

    let dir2 = dir2.as_ref();
    read_dir(dir1, |entry_path| {
        let entry_name = entry_path.file_name().unwrap();
        if entry_path.is_dir() {
            recursive_diff(&entry_path, &dir2.join(entry_name));
        } else {
            let path2 = dir2.join(entry_name);
            let file1 = read_file(&entry_path);
            let file2 = read_file(&path2);

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

pub fn read_dir<F: Fn(&Path)>(dir: impl AsRef<Path>, callback: F) {
    for entry in fs::read_dir(dir).unwrap() {
        callback(&entry.unwrap().path());
    }
}

/// Check that `haystack` does not contain `needle`. Panic otherwise.
pub fn assert_not_contains(haystack: &str, needle: &str) {
    if haystack.contains(needle) {
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", needle);
        panic!("needle was unexpectedly found in haystack");
    }
}

/// Implement common helpers for command wrappers. This assumes that the command wrapper is a struct
/// containing a `cmd: Command` field and a `output` function. The provided helpers are:
///
/// 1. Generic argument acceptors: `arg` and `args` (delegated to [`Command`]). These are intended
///    to be *fallback* argument acceptors, when specific helpers don't make sense. Prefer to add
///    new specific helper methods over relying on these generic argument providers.
/// 2. Environment manipulation methods: `env`, `env_remove` and `env_clear`: these delegate to
///    methods of the same name on [`Command`].
/// 3. Output and execution: `output`, `run` and `run_fail` are provided. `output` waits for the
///    command to finish running and returns the process's [`Output`]. `run` and `run_fail` are
///    higher-level convenience methods which waits for the command to finish running and assert
///    that the command successfully ran or failed as expected. Prefer `run` and `run_fail` when
///    possible.
///
/// Example usage:
///
/// ```ignore (illustrative)
/// struct CommandWrapper { cmd: Command } // <- required `cmd` field
///
/// impl CommandWrapper {
///     /// Get the [`Output`][::std::process::Output] of the finished process.
///     pub fn command_output(&mut self) -> Output { /* ... */ } // <- required `command_output()` method
/// }
///
/// crate::impl_common_helpers!(CommandWrapper);
///
/// impl CommandWrapper {
///     // ... additional specific helper methods
/// }
/// ```
///
/// [`Command`]: ::std::process::Command
/// [`Output`]: ::std::process::Output
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

            /// Clear all environmental variables.
            pub fn env_var(&mut self) -> &mut Self {
                self.cmd.env_clear();
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
                inspector(&self.cmd);
                self
            }

            /// Run the constructed command and assert that it is successfully run.
            #[track_caller]
            pub fn run(&mut self) -> ::std::process::Output {
                let caller_location = ::std::panic::Location::caller();
                let caller_line_number = caller_location.line();

                let output = self.command_output();
                if !output.status.success() {
                    handle_failed_output(&self.cmd, output, caller_line_number);
                }
                output
            }

            /// Run the constructed command and assert that it does not successfully run.
            #[track_caller]
            pub fn run_fail(&mut self) -> ::std::process::Output {
                let caller_location = ::std::panic::Location::caller();
                let caller_line_number = caller_location.line();

                let output = self.command_output();
                if output.status.success() {
                    handle_failed_output(&self.cmd, output, caller_line_number);
                }
                output
            }

            /// Set the path where the command will be run.
            pub fn current_dir<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
                self.cmd.current_dir(path);
                self
            }
        }
    };
}

pub(crate) use impl_common_helpers;
