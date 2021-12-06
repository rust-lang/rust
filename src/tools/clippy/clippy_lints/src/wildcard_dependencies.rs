use clippy_utils::{diagnostics::span_lint, is_lint_allowed};
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::DUMMY_SP;

use if_chain::if_chain;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for wildcard dependencies in the `Cargo.toml`.
    ///
    /// ### Why is this bad?
    /// [As the edition guide says](https://rust-lang-nursery.github.io/edition-guide/rust-2018/cargo-and-crates-io/crates-io-disallows-wildcard-dependencies.html),
    /// it is highly unlikely that you work with any possible version of your dependency,
    /// and wildcard dependencies would cause unnecessary breakage in the ecosystem.
    ///
    /// ### Example
    /// ```toml
    /// [dependencies]
    /// regex = "*"
    /// ```
    #[clippy::version = "1.32.0"]
    pub WILDCARD_DEPENDENCIES,
    cargo,
    "wildcard dependencies being used"
}

declare_lint_pass!(WildcardDependencies => [WILDCARD_DEPENDENCIES]);

impl LateLintPass<'_> for WildcardDependencies {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        if is_lint_allowed(cx, WILDCARD_DEPENDENCIES, CRATE_HIR_ID) {
            return;
        }

        let metadata = unwrap_cargo_metadata!(cx, WILDCARD_DEPENDENCIES, false);

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
