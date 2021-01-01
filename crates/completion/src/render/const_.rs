//! Renderer for `const` fields.

use hir::HasSource;
use syntax::{
    ast::{Const, NameOwner},
    display::const_label,
};

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    render::RenderContext,
};

pub(crate) fn render_const<'a>(
    ctx: RenderContext<'a>,
    const_: hir::Const,
) -> Option<CompletionItem> {
    ConstRender::new(ctx, const_).render()
}

#[derive(Debug)]
struct ConstRender<'a> {
    ctx: RenderContext<'a>,
    const_: hir::Const,
    ast_node: Const,
}

impl<'a> ConstRender<'a> {
    fn new(ctx: RenderContext<'a>, const_: hir::Const) -> ConstRender<'a> {
        #[allow(deprecated)]
        let ast_node = const_.source_old(ctx.db()).value;
        ConstRender { ctx, const_, ast_node }
    }

    fn render(self) -> Option<CompletionItem> {
        let name = self.name()?;
        let detail = self.detail();

        let item = CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), name)
            .kind(CompletionItemKind::Const)
            .set_documentation(self.ctx.docs(self.const_))
            .set_deprecated(self.ctx.is_deprecated(self.const_))
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
