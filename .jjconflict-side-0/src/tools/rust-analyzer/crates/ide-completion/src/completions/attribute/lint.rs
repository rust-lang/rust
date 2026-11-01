//! Completion for lints
use ide_db::{
    SymbolKind,
    documentation::Documentation,
    generated::lints::{CLIPPY_LINT_GROUPS, CLIPPY_LINTS, DEFAULT_LINTS, Lint, RUSTDOC_LINTS},
};
use syntax::ast;

use crate::{Completions, context::CompletionContext, item::CompletionItem};

pub(super) fn complete_lint(
    acc: &mut Completions,
    ctx: &CompletionContext<'_, '_>,
    is_qualified: bool,
    existing_lints: &[ast::Path],
) {
    let lints = (CLIPPY_LINT_GROUPS.iter().map(|g| &g.lint))
        .chain(DEFAULT_LINTS)
        .chain(CLIPPY_LINTS)
        .chain(RUSTDOC_LINTS);

    for &Lint { label, description, .. } in lints {
        // FIXME: change `Lint`'s label to not store a path in it but split the prefix off instead?
        let (qual, name) = match label.split_once("::") {
            Some((qual, name)) => (Some(qual), name),
            None => (None, label),
        };
        if qual.is_none() && is_qualified {
            // qualified completion requested, but this lint is unqualified
            continue;
        }
        let lint_already_annotated = existing_lints
            .iter()
            .filter_map(|path| {
                let q = path.qualifier();
                if q.as_ref().and_then(|it| it.qualifier()).is_some() {
                    return None;
                }
                Some((q.and_then(|it| it.as_single_name_ref()), path.segment()?.name_ref()?))
            })
            .any(|(q, name_ref)| {
                let qualifier_matches = match (q, qual) {
                    (None, None) => true,
                    (None, Some(_)) => false,
                    (Some(_), None) => false,
                    (Some(q), Some(ns)) => q.text() == ns,
                };
                qualifier_matches && name_ref.text() == name
            });
        if lint_already_annotated {
            continue;
        }
        let label = match qual {
            Some(qual) if !is_qualified => format!("{qual}::{name}"),
            _ => name.to_owned(),
        };
        let mut item =
            CompletionItem::new(SymbolKind::Attribute, ctx.source_range(), label, ctx.edition);
        item.documentation(Documentation::new_borrowed(description));
        item.add_to(acc, ctx.db)
    }
}
