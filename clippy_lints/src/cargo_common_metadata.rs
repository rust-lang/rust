// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! lint on missing cargo common metadata

use crate::rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::{ast::*, source_map::DUMMY_SP};
use crate::utils::span_lint;

use cargo_metadata;

/// **What it does:** Checks to see if all common metadata is defined in
/// `Cargo.toml`. See: https://rust-lang-nursery.github.io/api-guidelines/documentation.html#cargotoml-includes-all-common-metadata-c-metadata
///
/// **Why is this bad?** It will be more difficult for users to discover the
/// purpose of the crate, and key information related to it.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```toml
/// # This `Cargo.toml` is missing an authors field:
/// [package]
/// name = "clippy"
/// version = "0.0.212"
/// description = "A bunch of helpful lints to avoid common pitfalls in Rust"
/// repository = "https://github.com/rust-lang/rust-clippy"
/// readme = "README.md"
/// license = "MIT/Apache-2.0"
/// keywords = ["clippy", "lint", "plugin"]
/// categories = ["development-tools", "development-tools::cargo-plugins"]
/// ```
declare_clippy_lint! {
    pub CARGO_COMMON_METADATA,
    cargo,
    "common metadata is defined in `Cargo.toml`"
}

fn warning(cx: &EarlyContext<'_>, message: &str) {
    span_lint(cx, CARGO_COMMON_METADATA, DUMMY_SP, message);
}

fn missing_warning(cx: &EarlyContext<'_>, package: &cargo_metadata::Package, field: &str) {
    let message = format!("package `{}` is missing `{}` metadata", package.name, field);
    warning(cx, &message);
}

fn is_empty_str(value: &Option<String>) -> bool {
    match value {
        None => true,
        Some(value) if value.is_empty() => true,
        _ => false
    }
}

fn is_empty_vec(value: &[String]) -> bool {
    // This works because empty iterators return true
    value.iter().all(|v| v.is_empty())
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CARGO_COMMON_METADATA)
    }
}

impl EarlyLintPass for Pass {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &Crate) {
        let metadata = if let Ok(metadata) = cargo_metadata::metadata_deps(None, true) {
            metadata
        } else {
            warning(cx, "could not read cargo metadata");
            return;
        };

        for package in metadata.packages {
            if is_empty_vec(&package.authors) {
                missing_warning(cx, &package, "package.authors");
            }

            if is_empty_str(&package.description) {
                missing_warning(cx, &package, "package.description");
            }

            if is_empty_str(&package.license) {
                missing_warning(cx, &package, "package.license");
            }

            if is_empty_str(&package.repository) {
                missing_warning(cx, &package, "package.repository");
            }

            if is_empty_str(&package.readme) {
                missing_warning(cx, &package, "package.readme");
            }

            if is_empty_vec(&package.keywords) {
                missing_warning(cx, &package, "package.keywords");
            }

            if is_empty_vec(&package.categories) {
                missing_warning(cx, &package, "package.categories");
            }
        }
    }
}
