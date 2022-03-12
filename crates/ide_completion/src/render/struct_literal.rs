//! Renderer for `struct` literal.

use hir::{HasAttrs, Name, StructKind};
use syntax::SmolStr;

use crate::{
    render::compound::{
        format_literal_label, render_record, render_tuple, visible_fields, RenderedCompound,
    },
    render::RenderContext,
    CompletionItem, CompletionItemKind,
};

pub(crate) fn render_struct_literal(
    ctx: RenderContext<'_>,
    strukt: hir::Struct,
    path: Option<hir::ModPath>,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let _p = profile::span("render_struct_literal");

    let fields = strukt.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(&ctx, &fields, strukt)?;

    if fields_omitted {
        // If some fields are private you can't make `struct` literal.
        return None;
    }

    let name = local_name.unwrap_or_else(|| strukt.name(ctx.db())).to_smol_str();

    let rendered = render_literal(&ctx, path, &name, strukt.kind(ctx.db()), &visible_fields)?;

    Some(build_completion(&ctx, name, rendered, strukt.kind(ctx.db()), strukt))
}

fn build_completion(
    ctx: &RenderContext<'_>,
    name: SmolStr,
    rendered: RenderedCompound,
    kind: StructKind,
    def: impl HasAttrs + Copy,
) -> CompletionItem {
    let mut item = CompletionItem::new(
        CompletionItemKind::Snippet,
        ctx.source_range(),
        format_literal_label(&name, kind),
    );

    item.set_documentation(ctx.docs(def))
        .set_deprecated(ctx.is_deprecated(def))
        .detail(&rendered.detail)
        .set_relevance(ctx.completion_relevance());
    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, rendered.literal),
        None => item.insert_text(rendered.literal),
    };
    item.build()
}

fn render_literal(
    ctx: &RenderContext<'_>,
    path: Option<hir::ModPath>,
    name: &str,
    kind: StructKind,
    fields: &[hir::Field],
) -> Option<RenderedCompound> {
    let path_string;

    let qualified_name = if let Some(path) = path {
        path_string = path.to_string();
        &path_string
    } else {
        name
    };

    let mut rendered = match kind {
        StructKind::Tuple if ctx.snippet_cap().is_some() => {
            render_tuple(ctx.db(), ctx.snippet_cap(), fields, Some(qualified_name))
        }
        StructKind::Record => {
            render_record(ctx.db(), ctx.snippet_cap(), fields, Some(qualified_name))
        }
        _ => return None,
    };

    if ctx.snippet_cap().is_some() {
        rendered.literal.push_str("$0");
    }
    Some(rendered)
}
