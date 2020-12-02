//! Renderer for `enum` variants.

use hir::{HasAttrs, HirDisplay, ModPath, StructKind};
use itertools::Itertools;
use test_utils::mark;

use crate::{
    item::{CompletionItem, CompletionItemKind, CompletionKind, ImportToAdd},
    render::{builder_ext::Params, RenderContext},
};

pub(crate) fn render_enum_variant<'a>(
    ctx: RenderContext<'a>,
    import_to_add: Option<ImportToAdd>,
    local_name: Option<String>,
    variant: hir::EnumVariant,
    path: Option<ModPath>,
) -> CompletionItem {
    let _p = profile::span("render_enum_variant");
    EnumVariantRender::new(ctx, local_name, variant, path).render(import_to_add)
}

#[derive(Debug)]
struct EnumVariantRender<'a> {
    ctx: RenderContext<'a>,
    name: String,
    variant: hir::EnumVariant,
    path: Option<ModPath>,
    qualified_name: String,
    short_qualified_name: String,
    variant_kind: StructKind,
}

impl<'a> EnumVariantRender<'a> {
    fn new(
        ctx: RenderContext<'a>,
        local_name: Option<String>,
        variant: hir::EnumVariant,
        path: Option<ModPath>,
    ) -> EnumVariantRender<'a> {
        let name = local_name.unwrap_or_else(|| variant.name(ctx.db()).to_string());
        let variant_kind = variant.kind(ctx.db());

        let (qualified_name, short_qualified_name) = match &path {
            Some(path) => {
                let full = path.to_string();
                let short =
                    path.segments[path.segments.len().saturating_sub(2)..].iter().join("::");
                (full, short)
            }
            None => (name.to_string(), name.to_string()),
        };

        EnumVariantRender {
            ctx,
            name,
            variant,
            path,
            qualified_name,
            short_qualified_name,
            variant_kind,
        }
    }

    fn render(self, import_to_add: Option<ImportToAdd>) -> CompletionItem {
        let mut builder = CompletionItem::new(
            CompletionKind::Reference,
            self.ctx.source_range(),
            self.qualified_name.clone(),
        )
        .kind(CompletionItemKind::EnumVariant)
        .set_documentation(self.variant.docs(self.ctx.db()))
        .set_deprecated(self.ctx.is_deprecated(self.variant))
        .add_import(
            import_to_add,
            self.ctx.completion.config.should_resolve_additional_edits_immediately(),
        )
        .detail(self.detail());

        if self.variant_kind == StructKind::Tuple {
            mark::hit!(inserts_parens_for_tuple_enums);
            let params = Params::Anonymous(self.variant.fields(self.ctx.db()).len());
            builder =
                builder.add_call_parens(self.ctx.completion, self.short_qualified_name, params);
        } else if self.path.is_some() {
            builder = builder.lookup_by(self.short_qualified_name);
        }

        builder.build()
    }

    fn detail(&self) -> String {
        let detail_types = self
            .variant
            .fields(self.ctx.db())
            .into_iter()
            .map(|field| (field.name(self.ctx.db()), field.signature_ty(self.ctx.db())));

        match self.variant_kind {
            StructKind::Tuple | StructKind::Unit => format!(
                "({})",
                detail_types.map(|(_, t)| t.display(self.ctx.db()).to_string()).format(", ")
            ),
            StructKind::Record => format!(
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
    use test_utils::mark;

    use crate::test_utils::check_edit;

    #[test]
    fn inserts_parens_for_tuple_enums() {
        mark::check!(inserts_parens_for_tuple_enums);
        check_edit(
            "Some",
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main() -> Option<i32> {
    Som<|>
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
        check_edit(
            "Some",
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main(value: Option<i32>) {
    match value {
        Som<|>
    }
}
"#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main(value: Option<i32>) {
    match value {
        Some($0)
    }
}
"#,
        );
    }

    #[test]
    fn dont_duplicate_pattern_parens() {
        mark::check!(dont_duplicate_pattern_parens);
        check_edit(
            "Var",
            r#"
enum E { Var(i32) }
fn main() {
    match E::Var(92) {
        E::<|>(92) => (),
    }
}
"#,
            r#"
enum E { Var(i32) }
fn main() {
    match E::Var(92) {
        E::Var(92) => (),
    }
}
"#,
        );
    }
}
