//! Renderer for `struct` literal.

use hir::{db::HirDatabase, HasAttrs, HasVisibility, Name, StructKind};
use ide_db::helpers::SnippetCap;
use itertools::Itertools;

use crate::{item::CompletionKind, render::RenderContext, CompletionItem, CompletionItemKind};

pub(crate) fn render_struct_literal(
    ctx: RenderContext<'_>,
    strukt: hir::Struct,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let _p = profile::span("render_struct_literal");

    let fields = strukt.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(&ctx, &fields, strukt)?;

    if fields_omitted {
        // If some fields are private you can't make `struct` literal.
        return None;
    }

    let name = local_name.unwrap_or_else(|| strukt.name(ctx.db())).to_string();
    let literal = render_literal(&ctx, &name, strukt.kind(ctx.db()), &visible_fields)?;

    Some(build_completion(ctx, name, literal, strukt))
}

fn build_completion(
    ctx: RenderContext<'_>,
    name: String,
    literal: String,
    def: impl HasAttrs + Copy,
) -> CompletionItem {
    let mut item = CompletionItem::new(CompletionKind::Snippet, ctx.source_range(), name + " {…}");
    item.kind(CompletionItemKind::Snippet)
        .set_documentation(ctx.docs(def))
        .set_deprecated(ctx.is_deprecated(def))
        .detail(&literal);
    if let Some(snippet_cap) = ctx.snippet_cap() {
        item.insert_snippet(snippet_cap, literal);
    } else {
        item.insert_text(literal);
    };
    item.build()
}

fn render_literal(
    ctx: &RenderContext<'_>,
    name: &str,
    kind: StructKind,
    fields: &[hir::Field],
) -> Option<String> {
    let mut literal = match kind {
        StructKind::Tuple if ctx.snippet_cap().is_some() => render_tuple_as_literal(fields, name),
        StructKind::Record => render_record_as_literal(ctx.db(), ctx.snippet_cap(), fields, name),
        _ => return None,
    };

    if ctx.snippet_cap().is_some() {
        literal.push_str("$0");
    }
    Some(literal)
}

fn render_record_as_literal(
    db: &dyn HirDatabase,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: &str,
) -> String {
    let fields = fields.iter();
    if snippet_cap.is_some() {
        format!(
            "{name} {{ {} }}",
            fields
                .enumerate()
                .map(|(idx, field)| format!("{}: ${{{}:()}}", field.name(db), idx + 1))
                .format(", "),
            name = name
        )
    } else {
        format!(
            "{name} {{ {} }}",
            fields.map(|field| format!("{}: ()", field.name(db))).format(", "),
            name = name
        )
    }
}

fn render_tuple_as_literal(fields: &[hir::Field], name: &str) -> String {
    format!(
        "{name}({})",
        fields.iter().enumerate().map(|(idx, _)| format!("${}", idx + 1)).format(", "),
        name = name
    )
}

fn visible_fields(
    ctx: &RenderContext<'_>,
    fields: &[hir::Field],
    item: impl HasAttrs,
) -> Option<(Vec<hir::Field>, bool)> {
    let module = ctx.completion.scope.module()?;
    let n_fields = fields.len();
    let fields = fields
        .iter()
        .filter(|field| field.is_visible_from(ctx.db(), module))
        .copied()
        .collect::<Vec<_>>();

    let fields_omitted =
        n_fields - fields.len() > 0 || item.attrs(ctx.db()).by_key("non_exhaustive").exists();
    Some((fields, fields_omitted))
}
