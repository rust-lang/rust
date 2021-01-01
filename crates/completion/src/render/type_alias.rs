//! Renderer for type aliases.

use hir::HasSource;
use syntax::{
    ast::{NameOwner, TypeAlias},
    display::type_label,
};

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    render::RenderContext,
};

pub(crate) fn render_type_alias<'a>(
    ctx: RenderContext<'a>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    TypeAliasRender::new(ctx, type_alias).render()
}

#[derive(Debug)]
struct TypeAliasRender<'a> {
    ctx: RenderContext<'a>,
    type_alias: hir::TypeAlias,
    ast_node: TypeAlias,
}

impl<'a> TypeAliasRender<'a> {
    fn new(ctx: RenderContext<'a>, type_alias: hir::TypeAlias) -> TypeAliasRender<'a> {
        let ast_node = type_alias.source_old(ctx.db()).value;
        TypeAliasRender { ctx, type_alias, ast_node }
    }

    fn render(self) -> Option<CompletionItem> {
        let name = self.name()?;
        let detail = self.detail();

        let item = CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), name)
            .kind(CompletionItemKind::TypeAlias)
            .set_documentation(self.ctx.docs(self.type_alias))
            .set_deprecated(self.ctx.is_deprecated(self.type_alias))
            .detail(detail)
            .build();

        Some(item)
    }

    fn name(&self) -> Option<String> {
        self.ast_node.name().map(|name| name.text().to_string())
    }

    fn detail(&self) -> String {
        type_label(&self.ast_node)
    }
}
