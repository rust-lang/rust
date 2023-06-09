//! Renderer for `union` literals.

use hir::{HirDisplay, Name, StructKind};
use ide_db::SymbolKind;
use itertools::Itertools;

use crate::{
    render::{
        variant::{format_literal_label, format_literal_lookup, visible_fields},
        RenderContext,
    },
    CompletionItem, CompletionItemKind,
};

pub(crate) fn render_union_literal(
    ctx: RenderContext<'_>,
    un: hir::Union,
    path: Option<hir::ModPath>,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let name = local_name.unwrap_or_else(|| un.name(ctx.db()));

    let (qualified_name, escaped_qualified_name) = match path {
        Some(p) => (p.unescaped().display(ctx.db()).to_string(), p.display(ctx.db()).to_string()),
        None => {
            (name.unescaped().display(ctx.db()).to_string(), name.display(ctx.db()).to_string())
        }
    };
    let label = format_literal_label(&name.to_smol_str(), StructKind::Record, ctx.snippet_cap());
    let lookup = format_literal_lookup(&name.to_smol_str(), StructKind::Record);
    let mut item = CompletionItem::new(
        CompletionItemKind::SymbolKind(SymbolKind::Union),
        ctx.source_range(),
        label,
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
            fields.iter().map(|field| field.name(ctx.db()).to_smol_str()).format(",")
        )
    } else {
        format!(
            "{} {{ {} }}",
            escaped_qualified_name,
            fields.iter().format_with(", ", |field, f| {
                f(&format_args!("{}: ()", field.name(ctx.db()).display(ctx.db())))
            })
        )
    };

    let detail = format!(
        "{} {{ {}{} }}",
        qualified_name,
        fields.iter().format_with(", ", |field, f| {
            f(&format_args!(
                "{}: {}",
                field.name(ctx.db()).display(ctx.db()),
                field.ty(ctx.db()).display(ctx.db())
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
