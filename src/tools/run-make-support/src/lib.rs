//! `run-make-support` is a support library for run-make tests. It provides command wrappers and
//! convenience utility functions to help test writers reduce duplication. The support library
//! notably is built via cargo: this means that if your test wants some non-trivial utility, such
//! as `object` or `wasmparser`, they can be re-exported and be made available through this library.

mod command;
mod macros;
mod util;

pub mod ar;
pub mod artifact_names;
pub mod assertion_helpers;
pub mod diff;
pub mod env_checked;
pub mod external_deps;
pub mod fs_helpers;
pub mod fs_wrapper;
pub mod path_helpers;
pub mod run;
pub mod scoped_run;
pub mod targets;

use std::path::PathBuf;

// Re-exports of third-party library crates.
pub use bstr;
pub use gimli;
pub use object;
pub use regex;
pub use wasmparser;

// Re-exports of external dependencies.
pub use external_deps::{cc, clang, htmldocck, llvm, python, rustc, rustdoc};

// These rely on external dependencies.
pub use cc::{cc, extra_c_flags, extra_cxx_flags, Cc};
pub use clang::{clang, Clang};
pub use htmldocck::htmldocck;
pub use llvm::{
    llvm_ar, llvm_filecheck, llvm_objdump, llvm_profdata, llvm_readobj, LlvmAr, LlvmFilecheck,
    LlvmObjdump, LlvmProfdata, LlvmReadobj,
};
pub use python::python_command;
pub use rustc::{aux_build, bare_rustc, rustc, Rustc};
pub use rustdoc::{bare_rustdoc, rustdoc, Rustdoc};

/// [`ar`][mod@ar] currently uses the [ar][rust-ar] rust library, but that is subject to changes, we
/// may switch to `llvm-ar` subject to experimentation.
///
/// [rust-ar]: https://github.com/mdsteele/rust-ar
pub use ar::ar;

/// [`diff`][mod@diff] is implemented in terms of the [similar] library.
///
/// [similar]: https://github.com/mitsuhiko/similar
pub use diff::{diff, Diff};

/// Panic-on-fail [`std::env::var`] and [`std::env::var_os`] wrappers.
pub use env_checked::{env_var, env_var_os};

/// Convenience helpers for running binaries and other commands.
pub use run::{cmd, run, run_fail, run_with_args};

/// Helpers for checking target information.
pub use targets::{is_darwin, is_msvc, is_windows, target, uname};

/// Helpers for building names of output artifacts that are potentially target-specific.
pub use artifact_names::{
    bin_name, dynamic_lib_extension, dynamic_lib_name, rust_lib_name, static_lib_name,
};

/// Path-related helpers.
pub use path_helpers::{cwd, cygpath_windows, path, source_root};

/// Helpers for common fs operations.
pub use fs_helpers::{copy_dir_all, create_symlink, read_dir};

/// Helpers for scoped test execution where certain properties are attempted to be maintained.
pub use scoped_run::{run_in_tmpdir, test_while_readonly};

pub use assertion_helpers::{
    assert_contains, assert_equals, assert_not_contains, assert_recursive_eq,
    count_regex_matches_in_files_with_extension, filename_not_in_denylist, has_extension,
    has_prefix, has_suffix, invalid_utf8_contains, invalid_utf8_not_contains, not_contains,
    shallow_find_files,
};

use command::Command;

/// Builds a static lib (`.lib` on Windows MSVC and `.a` for the rest) with the given name.
#[track_caller]
pub fn build_native_static_lib(lib_name: &str) -> PathBuf {
    let obj_file = if is_msvc() { format!("{lib_name}") } else { format!("{lib_name}.o") };
    let src = format!("{lib_name}.c");
    let lib_path = static_lib_name(lib_name);
    if is_msvc() {
        cc().arg("-c").out_exe(&obj_file).input(src).run();
    } else {
        cc().arg("-v").arg("-c").out_exe(&obj_file).input(src).run();
    };
    let obj_file = if is_msvc() {
        PathBuf::from(format!("{lib_name}.obj"))
    } else {
        PathBuf::from(format!("{lib_name}.o"))
    };
    llvm_ar().obj_to_ar().output_input(&lib_path, &obj_file).run();
    path(lib_path)
}

/// Set the runtime library path as needed for running the host rustc/rustdoc/etc.
pub fn set_host_rpath(cmd: &mut Command) {
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(cwd());
        paths.push(PathBuf::from(env_var("HOST_RPATH_DIR")));
        for p in std::env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        std::env::join_paths(paths.iter()).unwrap()
    });
}
