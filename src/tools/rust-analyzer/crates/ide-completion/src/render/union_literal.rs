//! Renderer for `union` literals.

use hir::{HirDisplay, Name, StructKind};
use ide_db::SymbolKind;
use itertools::Itertools;
use syntax::ToSmolStr;

use crate::{
    CompletionItem, CompletionItemKind,
    render::{
        RenderContext,
        variant::{format_literal_label, format_literal_lookup, visible_fields},
    },
};

pub(crate) fn render_union_literal(
    ctx: RenderContext<'_>,
    un: hir::Union,
    path: Option<hir::ModPath>,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let name = local_name.unwrap_or_else(|| un.name(ctx.db()));

    let (qualified_name, escaped_qualified_name) = match path {
        Some(p) => (
            p.display_verbatim(ctx.db()).to_smolstr(),
            p.display(ctx.db(), ctx.completion.edition).to_smolstr(),
        ),
        None => (
            name.as_str().to_smolstr(),
            name.display(ctx.db(), ctx.completion.edition).to_smolstr(),
        ),
    };
    let label = format_literal_label(
        &name.display_no_db(ctx.completion.edition).to_smolstr(),
        StructKind::Record,
        ctx.snippet_cap(),
    );
    let lookup = format_literal_lookup(
        &name.display_no_db(ctx.completion.edition).to_smolstr(),
        StructKind::Record,
    );
    let mut item = CompletionItem::new(
        CompletionItemKind::SymbolKind(SymbolKind::Union),
        ctx.source_range(),
        label,
        ctx.completion.edition,
    );

    item.lookup_by(lookup);

    let fields = un.fields(ctx.db());
    let (fields, fields_omitted) = visible_fields(ctx.completion, &fields, un)?;

    if fields.is_empty() {
        return None;
    }

    let literal = if ctx.snippet_cap().is_some() {
        format!(
            "{} {{ ${{1|{}|}}: ${{2:()}} }}$0",
            escaped_qualified_name,
            fields
                .iter()
                .map(|field| field
                    .name(ctx.db())
                    .display_no_db(ctx.completion.edition)
                    .to_smolstr())
                .format(",")
        )
    } else {
        format!(
            "{} {{ {} }}",
            escaped_qualified_name,
            fields.iter().format_with(", ", |field, f| {
                f(&format_args!(
                    "{}: ()",
                    field.name(ctx.db()).display(ctx.db(), ctx.completion.edition)
                ))
            })
        )
    };

    let detail = format!(
        "{} {{ {}{} }}",
        qualified_name,
        fields.iter().format_with(", ", |field, f| {
            f(&format_args!(
                "{}: {}",
                field.name(ctx.db()).display(ctx.db(), ctx.completion.edition),
                field.ty(ctx.db()).display(ctx.db(), ctx.completion.display_target)
            ))
        }),
        if fields_omitted { ", .." } else { "" }
    );

    item.set_documentation(ctx.docs(un))
        .set_deprecated(ctx.is_deprecated(un))
        .detail(detail)
        .set_relevance(ctx.completion_relevance());

    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, literal).trigger_call_info(),
        None => item.insert_text(literal),
    };

    Some(item.build(ctx.db()))
}
