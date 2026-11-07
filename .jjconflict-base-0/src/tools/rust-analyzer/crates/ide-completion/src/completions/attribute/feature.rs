//! Completion for features
use ide_db::{
    SymbolKind,
    documentation::Documentation,
    generated::lints::{FEATURES, Lint},
};
use syntax::ast;

use crate::{Completions, context::CompletionContext, item::CompletionItem};

pub(super) fn complete_feature(
    acc: &mut Completions,
    ctx: &CompletionContext<'_, '_>,
    existing_features: &[ast::Path],
) {
    for &Lint { label, description, .. } in FEATURES {
        let feature_already_annotated = existing_features
            .iter()
            .filter_map(|p| p.as_single_name_ref())
            .any(|n| n.text() == label);
        if feature_already_annotated {
            continue;
        }

        let mut item =
            CompletionItem::new(SymbolKind::Attribute, ctx.source_range(), label, ctx.edition);
        item.documentation(Documentation::new_borrowed(description));
        item.add_to(acc, ctx.db)
    }
}
