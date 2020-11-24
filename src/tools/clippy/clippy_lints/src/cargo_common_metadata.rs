//! lint on missing cargo common metadata

use std::path::PathBuf;

use crate::utils::{run_lints, span_lint};
use rustc_hir::{hir_id::CRATE_HIR_ID, Crate};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::DUMMY_SP;

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
    /// license = "MIT OR Apache-2.0"
    /// keywords = ["clippy", "lint", "plugin"]
    /// categories = ["development-tools", "development-tools::cargo-plugins"]
    /// ```
    ///
    /// Should include an authors field like:
    ///
    /// ```toml
    /// # This `Cargo.toml` includes all common metadata
    /// [package]
    /// name = "clippy"
    /// version = "0.0.212"
    /// authors = ["Someone <someone@rust-lang.org>"]
    /// description = "A bunch of helpful lints to avoid common pitfalls in Rust"
    /// repository = "https://github.com/rust-lang/rust-clippy"
    /// readme = "README.md"
    /// license = "MIT OR Apache-2.0"
    /// keywords = ["clippy", "lint", "plugin"]
    /// categories = ["development-tools", "development-tools::cargo-plugins"]
    /// ```
    pub CARGO_COMMON_METADATA,
    cargo,
    "common metadata is defined in `Cargo.toml`"
}

fn missing_warning(cx: &LateContext<'_>, package: &cargo_metadata::Package, field: &str) {
    let message = format!("package `{}` is missing `{}` metadata", package.name, field);
    span_lint(cx, CARGO_COMMON_METADATA, DUMMY_SP, &message);
}

fn is_empty_str(value: &Option<String>) -> bool {
    value.as_ref().map_or(true, String::is_empty)
}

fn is_empty_path(value: &Option<PathBuf>) -> bool {
    value.as_ref().and_then(|x| x.to_str()).map_or(true, str::is_empty)
}

fn is_empty_vec(value: &[String]) -> bool {
    // This works because empty iterators return true
    value.iter().all(String::is_empty)
}

declare_lint_pass!(CargoCommonMetadata => [CARGO_COMMON_METADATA]);

impl LateLintPass<'_> for CargoCommonMetadata {
    fn check_crate(&mut self, cx: &LateContext<'_>, _: &Crate<'_>) {
        if !run_lints(cx, &[CARGO_COMMON_METADATA], CRATE_HIR_ID) {
            return;
        }

        let metadata = unwrap_cargo_metadata!(cx, CARGO_COMMON_METADATA, false);

        for package in metadata.packages {
            if is_empty_vec(&package.authors) {
                missing_warning(cx, &package, "package.authors");
            }

            if is_empty_str(&package.description) {
                missing_warning(cx, &package, "package.description");
            }

            if is_empty_str(&package.license) && is_empty_path(&package.license_file) {
                missing_warning(cx, &package, "either package.license or package.license_file");
            }

            if is_empty_str(&package.repository) {
                missing_warning(cx, &package, "package.repository");
            }

            if is_empty_path(&package.readme) {
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
