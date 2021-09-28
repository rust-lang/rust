//! lint on missing cargo common metadata

use clippy_utils::{diagnostics::span_lint, is_lint_allowed};
use rustc_hir::{hir_id::CRATE_HIR_ID, Crate};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::DUMMY_SP;

declare_clippy_lint! {
    /// ### What it does
    /// Checks to see if all common metadata is defined in
    /// `Cargo.toml`. See: https://rust-lang-nursery.github.io/api-guidelines/documentation.html#cargotoml-includes-all-common-metadata-c-metadata
    ///
    /// ### Why is this bad?
    /// It will be more difficult for users to discover the
    /// purpose of the crate, and key information related to it.
    ///
    /// ### Example
    /// ```toml
    /// # This `Cargo.toml` is missing a description field:
    /// [package]
    /// name = "clippy"
    /// version = "0.0.212"
    /// repository = "https://github.com/rust-lang/rust-clippy"
    /// readme = "README.md"
    /// license = "MIT OR Apache-2.0"
    /// keywords = ["clippy", "lint", "plugin"]
    /// categories = ["development-tools", "development-tools::cargo-plugins"]
    /// ```
    ///
    /// Should include a description field like:
    ///
    /// ```toml
    /// # This `Cargo.toml` includes all common metadata
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
    pub CARGO_COMMON_METADATA,
    cargo,
    "common metadata is defined in `Cargo.toml`"
}

#[derive(Copy, Clone, Debug)]
pub struct CargoCommonMetadata {
    ignore_publish: bool,
}

impl CargoCommonMetadata {
    pub fn new(ignore_publish: bool) -> Self {
        Self { ignore_publish }
    }
}

impl_lint_pass!(CargoCommonMetadata => [
    CARGO_COMMON_METADATA
]);

fn missing_warning(cx: &LateContext<'_>, package: &cargo_metadata::Package, field: &str) {
    let message = format!("package `{}` is missing `{}` metadata", package.name, field);
    span_lint(cx, CARGO_COMMON_METADATA, DUMMY_SP, &message);
}

fn is_empty_str<T: AsRef<std::ffi::OsStr>>(value: &Option<T>) -> bool {
    value.as_ref().map_or(true, |s| s.as_ref().is_empty())
}

fn is_empty_vec(value: &[String]) -> bool {
    // This works because empty iterators return true
    value.iter().all(String::is_empty)
}

impl LateLintPass<'_> for CargoCommonMetadata {
    fn check_crate(&mut self, cx: &LateContext<'_>, _: &Crate<'_>) {
        if is_lint_allowed(cx, CARGO_COMMON_METADATA, CRATE_HIR_ID) {
            return;
        }

        let metadata = unwrap_cargo_metadata!(cx, CARGO_COMMON_METADATA, false);

        for package in metadata.packages {
            // only run the lint if publish is `None` (`publish = true` or skipped entirely)
            // or if the vector isn't empty (`publish = ["something"]`)
            if package.publish.as_ref().filter(|publish| publish.is_empty()).is_none() || self.ignore_publish {
                if is_empty_str(&package.description) {
                    missing_warning(cx, &package, "package.description");
                }

                if is_empty_str(&package.license) && is_empty_str(&package.license_file) {
                    missing_warning(cx, &package, "either package.license or package.license_file");
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
}
