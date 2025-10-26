//! Renderer for `const` fields.

use hir::{AsAssocItem, HirDisplay};
use ide_db::SymbolKind;
use syntax::ToSmolStr;

use crate::{item::CompletionItem, render::RenderContext};

pub(crate) fn render_const(ctx: RenderContext<'_>, const_: hir::Const) -> Option<CompletionItem> {
    let _p = tracing::info_span!("render_const").entered();
    render(ctx, const_)
}

fn render(ctx: RenderContext<'_>, const_: hir::Const) -> Option<CompletionItem> {
    let db = ctx.db();
    let name = const_.name(db)?;
    let (name, escaped_name) =
        (name.as_str().to_smolstr(), name.display(db, ctx.completion.edition).to_smolstr());
    let detail = const_.display(db, ctx.completion.display_target).to_string();

    let mut item =
        CompletionItem::new(SymbolKind::Const, ctx.source_range(), name, ctx.completion.edition);
    item.set_documentation(ctx.docs(const_))
        .set_deprecated(ctx.is_deprecated(const_) || ctx.is_deprecated_assoc_item(const_))
        .detail(detail)
        .set_relevance(ctx.completion_relevance());

    if let Some(actm) = const_.as_assoc_item(db)
        && let Some(trt) = actm.container_or_implemented_trait(db)
    {
        item.trait_name(trt.name(db).display_no_db(ctx.completion.edition).to_smolstr());
    }
    item.insert_text(escaped_name);

    Some(item.build(ctx.db()))
}
