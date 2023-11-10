use cargo_metadata::Metadata;
use clippy_utils::diagnostics::span_lint;
use if_chain::if_chain;
use rustc_lint::LateContext;
use rustc_span::source_map::DUMMY_SP;

use super::WILDCARD_DEPENDENCIES;

pub(super) fn check(cx: &LateContext<'_>, metadata: &Metadata) {
    for dep in &metadata.packages[0].dependencies {
        // VersionReq::any() does not work
        if let Ok(wildcard_ver) = semver::VersionReq::parse("*")
            && let Some(ref source) = dep.source
            && !source.starts_with("git")
            && dep.req == wildcard_ver
        {
            span_lint(
                cx,
                WILDCARD_DEPENDENCIES,
                DUMMY_SP,
                &format!("wildcard dependency for `{}`", dep.name),
            );
        }
    }
}
