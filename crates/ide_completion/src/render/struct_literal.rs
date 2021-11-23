//! Renderer for `struct` literal.

use hir::{db::HirDatabase, HasAttrs, HasVisibility, Name, StructKind};
use ide_db::helpers::SnippetCap;
use itertools::Itertools;
use syntax::SmolStr;

use crate::{render::RenderContext, CompletionItem, CompletionItemKind};

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

    let literal = render_literal(&ctx, path, &name, strukt.kind(ctx.db()), &visible_fields)?;

    Some(build_completion(ctx, name, literal, strukt))
}

fn build_completion(
    ctx: RenderContext<'_>,
    name: SmolStr,
    literal: String,
    def: impl HasAttrs + Copy,
) -> CompletionItem {
    let mut item = CompletionItem::new(
        CompletionItemKind::Snippet,
        ctx.source_range(),
        SmolStr::from_iter([&name, " {…}"]),
    );
    item.set_documentation(ctx.docs(def)).set_deprecated(ctx.is_deprecated(def)).detail(&literal);
    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, literal),
        None => item.insert_text(literal),
    };
    item.build()
}

fn render_literal(
    ctx: &RenderContext<'_>,
    path: Option<hir::ModPath>,
    name: &str,
    kind: StructKind,
    fields: &[hir::Field],
) -> Option<String> {
    let path_string;

    let qualified_name = if let Some(path) = path {
        path_string = path.to_string();
        &path_string
    } else {
        name
    };

    let mut literal = match kind {
        StructKind::Tuple if ctx.snippet_cap().is_some() => {
            render_tuple_as_literal(fields, qualified_name)
        }
        StructKind::Record => {
            render_record_as_literal(ctx.db(), ctx.snippet_cap(), fields, qualified_name)
        }
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
