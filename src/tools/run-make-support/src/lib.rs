//! `run-make-support` is a support library for run-make tests. It provides command wrappers and
//! convenience utility functions to help test writers reduce duplication. The support library
//! notably is built via cargo: this means that if your test wants some non-trivial utility, such
//! as `object` or `wasmparser`, they can be re-exported and be made available through this library.

// We want to control use declaration ordering and spacing (and preserve use group comments), so
// skip rustfmt on this file.
#![cfg_attr(rustfmt, rustfmt::skip)]

mod command;
mod macros;
mod util;

pub mod artifact_names;
pub mod assertion_helpers;
pub mod diff;
pub mod env;
pub mod external_deps;
pub mod path_helpers;
pub mod run;
pub mod scoped_run;
pub mod string;
pub mod targets;

// Internally we call our fs-related support module as `fs`, but re-export its content as `rfs`
// to tests to avoid colliding with commonly used `use std::fs;`.
mod fs;

/// [`std::fs`] wrappers and assorted filesystem-related helpers. Public to tests as `rfs` to not be
/// confused with [`std::fs`].
pub mod rfs {
    pub use crate::fs::*;
}

// Re-exports of third-party library crates.
pub use bstr;
pub use gimli;
pub use object;
pub use regex;
pub use wasmparser;

// Re-exports of external dependencies.
pub use external_deps::{c_build, cc, clang, htmldocck, llvm, python, rustc, rustdoc};

// These rely on external dependencies.
pub use c_build::{build_native_dynamic_lib, build_native_static_lib};
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

/// [`diff`][mod@diff] is implemented in terms of the [similar] library.
///
/// [similar]: https://github.com/mitsuhiko/similar
pub use diff::{diff, Diff};

/// Panic-on-fail [`std::env::var`] and [`std::env::var_os`] wrappers.
pub use env::{env_var, env_var_os};

/// Convenience helpers for running binaries and other commands.
pub use run::{cmd, run, run_fail, run_with_args};

/// Helpers for checking target information.
pub use targets::{is_darwin, is_msvc, is_windows, llvm_components_contain, target, uname};

/// Helpers for building names of output artifacts that are potentially target-specific.
pub use artifact_names::{
    bin_name, dynamic_lib_extension, dynamic_lib_name, rust_lib_name, static_lib_name,
};

/// Path-related helpers.
pub use path_helpers::{
    cwd, filename_not_in_denylist, has_extension, has_prefix, has_suffix, not_contains, path,
    shallow_find_files, source_root,
};

/// Helpers for scoped test execution where certain properties are attempted to be maintained.
pub use scoped_run::{run_in_tmpdir, test_while_readonly};

pub use assertion_helpers::{
    assert_contains, assert_dirs_are_equal, assert_equals, assert_not_contains,
};

pub use string::{
    count_regex_matches_in_files_with_extension, invalid_utf8_contains, invalid_utf8_not_contains,
};
