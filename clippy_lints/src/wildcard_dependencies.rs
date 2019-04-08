use crate::utils::span_lint;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::{ast::*, source_map::DUMMY_SP};

use cargo_metadata;
use if_chain::if_chain;
use semver;

declare_clippy_lint! {
    /// **What it does:** Checks for wildcard dependencies in the `Cargo.toml`.
    ///
    /// **Why is this bad?** [As the edition guide says](https://rust-lang-nursery.github.io/edition-guide/rust-2018/cargo-and-crates-io/crates-io-disallows-wildcard-dependencies.html),
    /// it is highly unlikely that you work with any possible version of your dependency,
    /// and wildcard dependencies would cause unnecessary breakage in the ecosystem.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```toml
    /// [dependencies]
    /// regex = "*"
    /// ```
    pub WILDCARD_DEPENDENCIES,
    cargo,
    "wildcard dependencies being used"
}

declare_lint_pass!(WildcardDependencies => [WILDCARD_DEPENDENCIES]);

impl EarlyLintPass for WildcardDependencies {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &Crate) {
        let metadata = if let Ok(metadata) = cargo_metadata::MetadataCommand::new().no_deps().exec() {
            metadata
        } else {
            span_lint(cx, WILDCARD_DEPENDENCIES, DUMMY_SP, "could not read cargo metadata");
            return;
        };

        for dep in &metadata.packages[0].dependencies {
            // VersionReq::any() does not work
            if_chain! {
                if let Ok(wildcard_ver) = semver::VersionReq::parse("*");
                if let Some(ref source) = dep.source;
                if !source.starts_with("git");
                if dep.req == wildcard_ver;
                then {
                    span_lint(
                        cx,
                        WILDCARD_DEPENDENCIES,
                        DUMMY_SP,
                        &format!("wildcard dependency for `{}`", dep.name),
                    );
                }
            }
        }
    }
}
