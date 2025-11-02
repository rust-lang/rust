//! Injected inlay hints for `#[rust_analyzer::rust_fixture]`.

use hir::{EditionedFileId, Semantics};
use ide_db::{RootDatabase, impl_empty_upmap_from_ra_fixture, ra_fixture::UpmapFromRaFixture};
use syntax::{AstToken, ast};

use crate::{Analysis, InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind, InlayTooltip};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    file_id: EditionedFileId,
    config: &InlayHintsConfig<'_>,
    literal: ast::Literal,
) -> Option<()> {
    let file_id = file_id.file_id(sema.db);
    let literal = ast::String::cast(literal.token())?;
    let (analysis, fixture_analysis) =
        Analysis::from_ra_fixture(sema, literal.clone(), &literal, config.minicore)?;
    for virtual_file_id in fixture_analysis.files() {
        acc.extend(
            analysis
                .inlay_hints(config, virtual_file_id, None)
                .ok()?
                .upmap_from_ra_fixture(&fixture_analysis, virtual_file_id, file_id)
                .ok()?,
        );
    }
    Some(())
}

impl_empty_upmap_from_ra_fixture!(InlayHintPosition, InlayKind, InlayTooltip);
