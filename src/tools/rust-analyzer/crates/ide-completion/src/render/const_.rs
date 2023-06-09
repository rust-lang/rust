//! Renderer for `const` fields.

use hir::{AsAssocItem, HirDisplay};
use ide_db::SymbolKind;

use crate::{item::CompletionItem, render::RenderContext};

pub(crate) fn render_const(ctx: RenderContext<'_>, const_: hir::Const) -> Option<CompletionItem> {
    let _p = profile::span("render_const");
    render(ctx, const_)
}

fn render(ctx: RenderContext<'_>, const_: hir::Const) -> Option<CompletionItem> {
    let db = ctx.db();
    let name = const_.name(db)?;
    let (name, escaped_name) = (name.unescaped().to_smol_str(), name.to_smol_str());
    let detail = const_.display(db).to_string();

    let mut item = CompletionItem::new(SymbolKind::Const, ctx.source_range(), name);
    item.set_documentation(ctx.docs(const_))
        .set_deprecated(ctx.is_deprecated(const_) || ctx.is_deprecated_assoc_item(const_))
        .detail(detail)
        .set_relevance(ctx.completion_relevance());

    if let Some(actm) = const_.as_assoc_item(db) {
        if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
            item.trait_name(trt.name(db).to_smol_str());
        }
    }
    item.insert_text(escaped_name);

    Some(item.build(ctx.db()))
}
