//! Renderer for type aliases.

use hir::{AsAssocItem, HirDisplay};
use ide_db::SymbolKind;
use syntax::{SmolStr, ToSmolStr};

use crate::{item::CompletionItem, render::RenderContext};

pub(crate) fn render_type_alias(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    let _p = tracing::info_span!("render_type_alias").entered();
    render(ctx, type_alias, false)
}

pub(crate) fn render_type_alias_with_eq(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    let _p = tracing::info_span!("render_type_alias_with_eq").entered();
    render(ctx, type_alias, true)
}

fn render(
    ctx: RenderContext<'_>,
    type_alias: hir::TypeAlias,
    with_eq: bool,
) -> Option<CompletionItem> {
    let db = ctx.db();

    let name = type_alias.name(db);
    let (name, escaped_name) = if with_eq {
        (
            SmolStr::from_iter([&name.as_str().to_smolstr(), " = "]),
            SmolStr::from_iter([&name.display_no_db(ctx.completion.edition).to_smolstr(), " = "]),
        )
    } else {
        (name.as_str().to_smolstr(), name.display_no_db(ctx.completion.edition).to_smolstr())
    };
    let detail = type_alias.display(db, ctx.completion.display_target).to_string();

    let mut item = CompletionItem::new(
        SymbolKind::TypeAlias,
        ctx.source_range(),
        name,
        ctx.completion.edition,
    );
    item.set_documentation(ctx.docs(type_alias))
        .set_deprecated(ctx.is_deprecated(type_alias) || ctx.is_deprecated_assoc_item(type_alias))
        .detail(detail)
        .set_relevance(ctx.completion_relevance());

    if let Some(actm) = type_alias.as_assoc_item(db)
        && let Some(trt) = actm.container_or_implemented_trait(db)
    {
        item.trait_name(trt.name(db).display_no_db(ctx.completion.edition).to_smolstr());
    }
    item.insert_text(escaped_name);

    Some(item.build(ctx.db()))
}
