//! Renderer for `enum` variants.

use hir::{db::HirDatabase, HasAttrs, HirDisplay, StructKind};
use ide_db::SymbolKind;
use itertools::Itertools;
use syntax::SmolStr;

use crate::{
    item::{CompletionItem, ImportEdit},
    render::{builder_ext::Params, compute_ref_match, compute_type_match, RenderContext},
    CompletionRelevance,
};

pub(crate) fn render_variant(
    ctx: RenderContext<'_>,
    import_to_add: Option<ImportEdit>,
    local_name: Option<hir::Name>,
    variant: hir::Variant,
    path: Option<hir::ModPath>,
) -> CompletionItem {
    let _p = profile::span("render_enum_variant");
    render(ctx, local_name, variant, path, import_to_add)
}

fn render(
    ctx @ RenderContext { completion }: RenderContext<'_>,
    local_name: Option<hir::Name>,
    variant: hir::Variant,
    path: Option<hir::ModPath>,
    import_to_add: Option<ImportEdit>,
) -> CompletionItem {
    let db = completion.db;
    let name = local_name.unwrap_or_else(|| variant.name(db));
    let variant_kind = variant.kind(db);

    let (qualified_name, short_qualified_name, qualified) = match path {
        Some(path) => {
            let short = hir::ModPath::from_segments(
                hir::PathKind::Plain,
                path.segments().iter().skip(path.segments().len().saturating_sub(2)).cloned(),
            );
            (path, short, true)
        }
        None => (name.clone().into(), name.into(), false),
    };
    let qualified_name = qualified_name.to_string();
    let short_qualified_name: SmolStr = short_qualified_name.to_string().into();

    let mut item = CompletionItem::new(SymbolKind::Variant, ctx.source_range(), qualified_name);
    item.set_documentation(variant.docs(db))
        .set_deprecated(ctx.is_deprecated(variant))
        .detail(detail(db, variant, variant_kind));

    if let Some(import_to_add) = import_to_add {
        item.add_import(import_to_add);
    }

    if variant_kind == hir::StructKind::Tuple {
        cov_mark::hit!(inserts_parens_for_tuple_enums);
        let params = Params::Anonymous(variant.fields(db).len());
        item.add_call_parens(ctx.completion, short_qualified_name, params);
    } else if qualified {
        item.lookup_by(short_qualified_name);
    }

    let ty = variant.parent_enum(ctx.completion.db).ty(ctx.completion.db);
    item.set_relevance(CompletionRelevance {
        type_match: compute_type_match(ctx.completion, &ty),
        ..CompletionRelevance::default()
    });

    if let Some(ref_match) = compute_ref_match(ctx.completion, &ty) {
        item.ref_match(ref_match);
    }

    item.build()
}

fn detail(db: &dyn HirDatabase, variant: hir::Variant, variant_kind: StructKind) -> String {
    let detail_types = variant.fields(db).into_iter().map(|field| (field.name(db), field.ty(db)));

    match variant_kind {
        hir::StructKind::Tuple | hir::StructKind::Unit => {
            format!("({})", detail_types.format_with(", ", |(_, t), f| f(&t.display(db))))
        }
        hir::StructKind::Record => {
            format!(
                "{{{}}}",
                detail_types.format_with(", ", |(n, t), f| {
                    f(&n)?;
                    f(&": ")?;
                    f(&t.display(db))
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::check_edit;

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
