//! Renderer for `enum` variants.

use hir::{HasAttrs, StructKind};
use ide_db::SymbolKind;
use syntax::SmolStr;

use crate::{
    item::{CompletionItem, ImportEdit},
    render::{
        compound::{format_literal_label, render_record, render_tuple, RenderedCompound},
        compute_ref_match, compute_type_match, RenderContext,
    },
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
    ctx @ RenderContext { completion, .. }: RenderContext<'_>,
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

    let mut rendered = match variant_kind {
        StructKind::Tuple => {
            render_tuple(db, ctx.snippet_cap(), &variant.fields(db), Some(&qualified_name))
        }
        StructKind::Record => {
            render_record(db, ctx.snippet_cap(), &variant.fields(db), Some(&qualified_name))
        }
        StructKind::Unit => {
            RenderedCompound { literal: qualified_name.clone(), detail: qualified_name.clone() }
        }
    };

    if ctx.snippet_cap().is_some() {
        rendered.literal.push_str("$0");
    }

    let mut item = CompletionItem::new(
        SymbolKind::Variant,
        ctx.source_range(),
        format_literal_label(&qualified_name, variant_kind),
    );

    item.set_documentation(variant.docs(db))
        .set_deprecated(ctx.is_deprecated(variant))
        .detail(rendered.detail);

    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, rendered.literal),
        None => item.insert_text(rendered.literal),
    };

    if let Some(import_to_add) = import_to_add {
        item.add_import(import_to_add);
    }

    if qualified {
        item.lookup_by(short_qualified_name);
    }

    let ty = variant.parent_enum(completion.db).ty(completion.db);
    item.set_relevance(CompletionRelevance {
        type_match: compute_type_match(completion, &ty),
        ..ctx.completion_relevance()
    });

    if let Some(ref_match) = compute_ref_match(completion, &ty) {
        item.ref_match(ref_match);
    }

    item.build()
}
