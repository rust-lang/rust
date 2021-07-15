//! Completion for lints
use ide_db::helpers::generated_lints::Lint;
use syntax::ast;

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    Completions,
};

pub(super) fn complete_lint(
    acc: &mut Completions,
    ctx: &CompletionContext,
    derive_input: ast::TokenTree,
    lints_completions: &[Lint],
) {
    if let Some(existing_lints) = super::parse_comma_sep_input(derive_input) {
        for lint_completion in lints_completions
            .into_iter()
            .filter(|completion| !existing_lints.contains(completion.label))
        {
            let mut item = CompletionItem::new(
                CompletionKind::Attribute,
                ctx.source_range(),
                lint_completion.label,
            );
            item.kind(CompletionItemKind::Attribute)
                .documentation(hir::Documentation::new(lint_completion.description.to_owned()));
            item.add_to(acc)
        }
    }
}
