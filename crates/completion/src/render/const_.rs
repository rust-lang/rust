//! Renderer for `const` fields.

use hir::HasSource;
use ide_db::SymbolKind;
use syntax::{
    ast::{Const, NameOwner},
    display::const_label,
};

use crate::{
    item::{CompletionItem, CompletionKind},
    render::RenderContext,
};

pub(crate) fn render_const<'a>(
    ctx: RenderContext<'a>,
    const_: hir::Const,
) -> Option<CompletionItem> {
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
        let name = self.name()?;
        let detail = self.detail();

        let item = CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), name)
            .kind(SymbolKind::Const)
            .set_documentation(self.ctx.docs(self.const_))
            .set_deprecated(
                self.ctx.is_deprecated(self.const_)
                    || self.ctx.is_deprecated_assoc_item(self.const_),
            )
            .detail(detail)
            .build();

        Some(item)
    }

    fn name(&self) -> Option<String> {
        self.ast_node.name().map(|name| name.text().to_string())
    }

    fn detail(&self) -> String {
        const_label(&self.ast_node)
    }
}
