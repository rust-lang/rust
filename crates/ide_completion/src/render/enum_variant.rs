//! Renderer for `enum` variants.

use std::iter;

use hir::{HasAttrs, HirDisplay};
use ide_db::SymbolKind;
use itertools::Itertools;

use crate::{
    item::{CompletionItem, CompletionKind, ImportEdit},
    render::{builder_ext::Params, compute_ref_match, compute_type_match, RenderContext},
    CompletionRelevance,
};

pub(crate) fn render_variant<'a>(
    ctx: RenderContext<'a>,
    import_to_add: Option<ImportEdit>,
    local_name: Option<hir::Name>,
    variant: hir::Variant,
    path: Option<hir::ModPath>,
) -> CompletionItem {
    let _p = profile::span("render_enum_variant");
    EnumRender::new(ctx, local_name, variant, path).render(import_to_add)
}

#[derive(Debug)]
struct EnumRender<'a> {
    ctx: RenderContext<'a>,
    name: hir::Name,
    variant: hir::Variant,
    path: Option<hir::ModPath>,
    qualified_name: hir::ModPath,
    short_qualified_name: hir::ModPath,
    variant_kind: hir::StructKind,
}

impl<'a> EnumRender<'a> {
    fn new(
        ctx: RenderContext<'a>,
        local_name: Option<hir::Name>,
        variant: hir::Variant,
        path: Option<hir::ModPath>,
    ) -> EnumRender<'a> {
        let name = local_name.unwrap_or_else(|| variant.name(ctx.db()));
        let variant_kind = variant.kind(ctx.db());

        let (qualified_name, short_qualified_name) = match &path {
            Some(path) => {
                let short = hir::ModPath::from_segments(
                    hir::PathKind::Plain,
                    path.segments().iter().skip(path.segments().len().saturating_sub(2)).cloned(),
                );
                (path.clone(), short)
            }
            None => (
                hir::ModPath::from_segments(hir::PathKind::Plain, iter::once(name.clone())),
                hir::ModPath::from_segments(hir::PathKind::Plain, iter::once(name.clone())),
            ),
        };

        EnumRender { ctx, name, variant, path, qualified_name, short_qualified_name, variant_kind }
    }
    fn render(self, import_to_add: Option<ImportEdit>) -> CompletionItem {
        let mut item = CompletionItem::new(
            CompletionKind::Reference,
            self.ctx.source_range(),
            self.qualified_name.to_string(),
        );
        item.kind(SymbolKind::Variant)
            .set_documentation(self.variant.docs(self.ctx.db()))
            .set_deprecated(self.ctx.is_deprecated(self.variant))
            .add_import(import_to_add)
            .detail(self.detail());

        if self.variant_kind == hir::StructKind::Tuple {
            cov_mark::hit!(inserts_parens_for_tuple_enums);
            let params = Params::Anonymous(self.variant.fields(self.ctx.db()).len());
            item.add_call_parens(
                self.ctx.completion,
                self.short_qualified_name.to_string(),
                params,
            );
        } else if self.path.is_some() {
            item.lookup_by(self.short_qualified_name.to_string());
        }

        let ty = self.variant.parent_enum(self.ctx.completion.db).ty(self.ctx.completion.db);
        item.set_relevance(CompletionRelevance {
            type_match: compute_type_match(self.ctx.completion, &ty),
            ..CompletionRelevance::default()
        });

        if let Some(ref_match) = compute_ref_match(self.ctx.completion, &ty) {
            item.ref_match(ref_match);
        }

        item.build()
    }

    fn detail(&self) -> String {
        let detail_types = self
            .variant
            .fields(self.ctx.db())
            .into_iter()
            .map(|field| (field.name(self.ctx.db()), field.ty(self.ctx.db())));

        match self.variant_kind {
            hir::StructKind::Tuple | hir::StructKind::Unit => format!(
                "({})",
                detail_types.map(|(_, t)| t.display(self.ctx.db()).to_string()).format(", ")
            ),
            hir::StructKind::Record => format!(
                "{{ {} }}",
                detail_types
                    .map(|(n, t)| format!("{}: {}", n, t.display(self.ctx.db()).to_string()))
                    .format(", ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::check_edit;

    #[test]
    fn inserts_parens_for_tuple_enums() {
        cov_mark::check!(inserts_parens_for_tuple_enums);
        check_edit(
            "Some",
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main() -> Option<i32> {
    Som$0
}
"#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main() -> Option<i32> {
    Some($0)
}
"#,
        );
    }
}
