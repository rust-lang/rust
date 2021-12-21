//! Renderer for type aliases.

use hir::{AsAssocItem, HasSource};
use ide_db::SymbolKind;
use syntax::{ast::HasName, display::type_label};

use crate::{item::CompletionItem, render::RenderContext};

pub(crate) fn render_type_alias(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    let _p = profile::span("render_type_alias");
    render(ctx, type_alias, false)
}

pub(crate) fn render_type_alias_with_eq(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    let _p = profile::span("render_type_alias_with_eq");
    render(ctx, type_alias, true)
}

fn render(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
    with_eq: bool,
) -> Option<CompletionItem> {
    let db = ctx.db();

    // FIXME: This parses the file!
    let ast_node = type_alias.source(db)?.value;
    let name = ast_node.name().map(|name| {
        if with_eq {
            format!("{} = ", name.text())
        } else {
            name.text().to_string()
        }
    })?;
    let detail = type_label(&ast_node);

    let mut item = CompletionItem::new(SymbolKind::TypeAlias, ctx.source_range(), name.clone());
    item.set_documentation(ctx.docs(type_alias))
        .set_deprecated(ctx.is_deprecated(type_alias) || ctx.is_deprecated_assoc_item(type_alias))
        .detail(detail);

    if let Some(actm) = type_alias.as_assoc_item(db) {
        if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
            item.trait_name(trt.name(db).to_smol_str());
            item.insert_text(name);
        }
    }

    Some(item.build())
}
