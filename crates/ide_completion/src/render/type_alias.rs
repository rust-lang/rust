//! Renderer for type aliases.

use hir::{AsAssocItem, HasSource};
use ide_db::SymbolKind;
use syntax::{
    ast::{NameOwner, TypeAlias},
    display::type_label,
};

use crate::{
    item::{CompletionItem, CompletionKind},
    render::RenderContext,
};

pub(crate) fn render_type_alias<'a>(
    ctx: RenderContext<'a>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    TypeAliasRender::new(ctx, type_alias)?.render(false)
}

pub(crate) fn render_type_alias_with_eq<'a>(
    ctx: RenderContext<'a>,
    type_alias: hir::TypeAlias,
) -> Option<CompletionItem> {
    TypeAliasRender::new(ctx, type_alias)?.render(true)
}

#[derive(Debug)]
struct TypeAliasRender<'a> {
    ctx: RenderContext<'a>,
    type_alias: hir::TypeAlias,
    ast_node: TypeAlias,
}

impl<'a> TypeAliasRender<'a> {
    fn new(ctx: RenderContext<'a>, type_alias: hir::TypeAlias) -> Option<TypeAliasRender<'a>> {
        let ast_node = type_alias.source(ctx.db())?.value;
        Some(TypeAliasRender { ctx, type_alias, ast_node })
    }

    fn render(self, with_eq: bool) -> Option<CompletionItem> {
        let name = self.ast_node.name().map(|name| {
            if with_eq {
                format!("{} = ", name.text())
            } else {
                name.text().to_string()
            }
        })?;
        let detail = self.detail();

        let mut item =
            CompletionItem::new(CompletionKind::Reference, self.ctx.source_range(), name.clone());
        item.kind(SymbolKind::TypeAlias)
            .set_documentation(self.ctx.docs(self.type_alias))
            .set_deprecated(
                self.ctx.is_deprecated(self.type_alias)
                    || self.ctx.is_deprecated_assoc_item(self.type_alias),
            )
            .detail(detail);

        let db = self.ctx.db();
        if let Some(actm) = self.type_alias.as_assoc_item(db) {
            if let Some(trt) = actm.containing_trait_or_trait_impl(db) {
                item.trait_name(trt.name(db).to_string());
                item.insert_text(name.clone());
            }
        }

        Some(item.build())
    }

    fn detail(&self) -> String {
        type_label(&self.ast_node)
    }
}
