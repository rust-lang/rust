//! Renderer for `const` fields.

use hir::{AsAssocItem, HasSource};
use ide_db::SymbolKind;
use syntax::{ast::Const, display::const_label};

use crate::{item::CompletionItem, render::RenderContext};

pub(crate) fn render_const(ctx: RenderContext<'_>, const_: hir::Const) -> Option<CompletionItem> {
    let _p = profile::span("render_const");
    ConstRender::new(ctx, const_)?.render()
}

#[derive(Debug)]
struct ConstRender<'a> {
    ctx: RenderContext<'a>,
    const_: hir::Const,
    ast_node: Const,
}

impl<'a> ConstRender<'a> {
    fn new(ctx: RenderContext<'a>, const_: hir::Const) -> Option<ConstRender<'a>> {
        let ast_node = const_.source(ctx.db())?.value;
        Some(ConstRender { ctx, const_, ast_node })
    }

    fn render(self) -> Option<CompletionItem> {
        let name = self.const_.name(self.ctx.db())?.to_smol_str();
        let detail = self.detail();

        let mut item =
            CompletionItem::new(SymbolKind::Const, self.ctx.source_range(), name.clone());
        item.set_documentation(self.ctx.docs(self.const_))
            .set_deprecated(
                self.ctx.is_deprecated(self.const_)
                    || self.ctx.is_deprecated_assoc_item(self.const_),
            )
            .detail(detail);

        let db = self.ctx.db();
        if let Some(actm) = self.const_.as_assoc_item(db) {
            if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
                item.trait_name(trt.name(db).to_smol_str());
                item.insert_text(name);
            }
        }

        Some(item.build())
    }

    fn detail(&self) -> String {
        const_label(&self.ast_node)
    }
}
