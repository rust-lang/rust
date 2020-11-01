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

#[derive(Debug)]
pub(crate) struct ConstRender<'a> {
    ctx: RenderContext<'a>,
    const_: hir::Const,
    ast_node: Const,
}

impl<'a> ConstRender<'a> {
    pub(crate) fn new(ctx: RenderContext<'a>, const_: hir::Const) -> ConstRender<'a> {
        let ast_node = const_.source(ctx.db()).value;
        ConstRender { ctx, const_, ast_node }
    }

    pub(crate) fn render(self) -> Option<CompletionItem> {
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
