//! Renderer for `union` literals.

use hir::{HirDisplay, Name, StructKind};
use itertools::Itertools;

use crate::{
    render::{
        compound::{format_literal_label, visible_fields},
        RenderContext,
    },
    CompletionItem, CompletionItemKind,
};

pub(crate) fn render_union_literal(
    ctx: RenderContext,
    un: hir::Union,
    path: Option<hir::ModPath>,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let name = local_name.unwrap_or_else(|| un.name(ctx.db())).to_smol_str();

    let qualified_name = match path {
        Some(p) => p.to_string(),
        None => name.to_string(),
    };

    let mut item = CompletionItem::new(
        CompletionItemKind::Snippet,
        ctx.source_range(),
        format_literal_label(&name, StructKind::Record),
    );

    let fields = un.fields(ctx.db());
    let (fields, fields_omitted) = visible_fields(&ctx, &fields, un)?;

    if fields.is_empty() {
        return None;
    }

    let literal = if ctx.snippet_cap().is_some() {
        format!(
            "{} {{ ${{1|{}|}}: ${{2:()}} }}$0",
            qualified_name,
            fields.iter().map(|field| field.name(ctx.db())).format(",")
        )
    } else {
        format!(
            "{} {{ {} }}",
            qualified_name,
            fields
                .iter()
                .format_with(", ", |field, f| { f(&format_args!("{}: ()", field.name(ctx.db()))) })
        )
    };

    let detail = format!(
        "{} {{ {}{} }}",
        qualified_name,
        fields.iter().format_with(", ", |field, f| {
            f(&format_args!("{}: {}", field.name(ctx.db()), field.ty(ctx.db()).display(ctx.db())))
        }),
        if fields_omitted { ", .." } else { "" }
    );

    item.set_documentation(ctx.docs(un))
        .set_deprecated(ctx.is_deprecated(un))
        .detail(&detail)
        .set_relevance(ctx.completion_relevance());

    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, literal),
        None => item.insert_text(literal),
    };

    Some(item.build())
}
