//! lint on multiple versions of a crate being used

use crate::utils::{run_lints, span_lint};
use rustc_hir::{Crate, CRATE_HIR_ID};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::DUMMY_SP;

use itertools::Itertools;

declare_clippy_lint! {
    /// **What it does:** Checks to see if multiple versions of a crate are being
    /// used.
    ///
    /// **Why is this bad?** This bloats the size of targets, and can lead to
    /// confusing error messages when structs or traits are used interchangeably
    /// between different versions of a crate.
    ///
    /// **Known problems:** Because this can be caused purely by the dependencies
    /// themselves, it's not always possible to fix this issue.
    ///
    /// **Example:**
    /// ```toml
    /// # This will pull in both winapi v0.3.x and v0.2.x, triggering a warning.
    /// [dependencies]
    /// ctrlc = "=3.1.0"
    /// ansi_term = "=0.11.0"
    /// ```
    pub MULTIPLE_CRATE_VERSIONS,
    cargo,
    "multiple versions of the same crate being used"
}

declare_lint_pass!(MultipleCrateVersions => [MULTIPLE_CRATE_VERSIONS]);

impl LateLintPass<'_, '_> for MultipleCrateVersions {
    fn check_crate(&mut self, cx: &LateContext<'_, '_>, _: &Crate<'_>) {
        if !run_lints(cx, &[MULTIPLE_CRATE_VERSIONS], CRATE_HIR_ID) {
            return;
        }

        let metadata = if let Ok(metadata) = cargo_metadata::MetadataCommand::new().exec() {
            metadata
        } else {
            span_lint(cx, MULTIPLE_CRATE_VERSIONS, DUMMY_SP, "could not read cargo metadata");

            return;
        };

        let mut packages = metadata.packages;
        packages.sort_by(|a, b| a.name.cmp(&b.name));

        for (name, group) in &packages.into_iter().group_by(|p| p.name.clone()) {
            let group: Vec<cargo_metadata::Package> = group.collect();

            if group.len() > 1 {
                let versions = group.into_iter().map(|p| p.version).join(", ");

                span_lint(
                    cx,
                    MULTIPLE_CRATE_VERSIONS,
                    DUMMY_SP,
                    &format!("multiple versions for dependency `{}`: {}", name, versions),
                );
            }
        }
    }
}
