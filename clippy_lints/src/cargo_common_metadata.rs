//! lint on missing cargo common metadata

use crate::utils::span_lint;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::{ast::*, source_map::DUMMY_SP};

use cargo_metadata;

declare_clippy_lint! {
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
        _ => false,
    }
}

fn is_empty_vec(value: &[String]) -> bool {
    // This works because empty iterators return true
    value.iter().all(std::string::String::is_empty)
}

declare_lint_pass!(CargoCommonMetadata => [CARGO_COMMON_METADATA]);

impl EarlyLintPass for CargoCommonMetadata {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &Crate) {
        let metadata = if let Ok(metadata) = cargo_metadata::MetadataCommand::new().no_deps().exec() {
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
