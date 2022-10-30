//! Renderer for `enum` variants.

use hir::{db::HirDatabase, Documentation, HasAttrs, StructKind};
use ide_db::SymbolKind;

use crate::{
    context::{CompletionContext, PathCompletionCtx, PathKind},
    item::{Builder, CompletionItem},
    render::{
        compute_type_match,
        variant::{
            format_literal_label, format_literal_lookup, render_record_lit, render_tuple_lit,
            visible_fields, RenderedLiteral,
        },
        RenderContext,
    },
    CompletionItemKind, CompletionRelevance,
};

pub(crate) fn render_variant_lit(
    ctx: RenderContext<'_>,
    path_ctx: &PathCompletionCtx,
    local_name: Option<hir::Name>,
    variant: hir::Variant,
    path: Option<hir::ModPath>,
) -> Option<Builder> {
    let _p = profile::span("render_enum_variant");
    let db = ctx.db();

    let name = local_name.unwrap_or_else(|| variant.name(db));
    render(ctx, path_ctx, Variant::EnumVariant(variant), name, path)
}

pub(crate) fn render_struct_literal(
    ctx: RenderContext<'_>,
    path_ctx: &PathCompletionCtx,
    strukt: hir::Struct,
    path: Option<hir::ModPath>,
    local_name: Option<hir::Name>,
) -> Option<Builder> {
    let _p = profile::span("render_struct_literal");
    let db = ctx.db();

    let name = local_name.unwrap_or_else(|| strukt.name(db));
    render(ctx, path_ctx, Variant::Struct(strukt), name, path)
}

fn render(
    ctx @ RenderContext { completion, .. }: RenderContext<'_>,
    path_ctx: &PathCompletionCtx,
    thing: Variant,
    name: hir::Name,
    path: Option<hir::ModPath>,
) -> Option<Builder> {
    let db = completion.db;
    let mut kind = thing.kind(db);
    let should_add_parens = match &path_ctx {
        PathCompletionCtx { has_call_parens: true, .. } => false,
        PathCompletionCtx { kind: PathKind::Use | PathKind::Type { .. }, .. } => false,
        _ => true,
    };

    let fields = thing.fields(completion)?;
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
    let (qualified_name, escaped_qualified_name) =
        (qualified_name.unescaped().to_string(), qualified_name.to_string());
    let snippet_cap = ctx.snippet_cap();

    let mut rendered = match kind {
        StructKind::Tuple if should_add_parens => {
            render_tuple_lit(db, snippet_cap, &fields, &escaped_qualified_name)
        }
        StructKind::Record if should_add_parens => {
            render_record_lit(db, snippet_cap, &fields, &escaped_qualified_name)
        }
        _ => RenderedLiteral {
            literal: escaped_qualified_name.clone(),
            detail: escaped_qualified_name.clone(),
        },
    };

    if snippet_cap.is_some() {
        rendered.literal.push_str("$0");
    }

    // only show name in label if not adding parens
    if !should_add_parens {
        kind = StructKind::Unit;
    }
    let label = format_literal_label(&qualified_name, kind);
    let lookup = if qualified {
        format_literal_lookup(&short_qualified_name.to_string(), kind)
    } else {
        format_literal_lookup(&qualified_name, kind)
    };

    let mut item = CompletionItem::new(
        CompletionItemKind::SymbolKind(thing.symbol_kind()),
        ctx.source_range(),
        label,
    );

    item.lookup_by(lookup);
    item.detail(rendered.detail);

    match snippet_cap {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, rendered.literal),
        None => item.insert_text(rendered.literal),
    };

    item.set_documentation(thing.docs(db)).set_deprecated(thing.is_deprecated(&ctx));

    let ty = thing.ty(db);
    item.set_relevance(CompletionRelevance {
        type_match: compute_type_match(ctx.completion, &ty),
        ..ctx.completion_relevance()
    });

    super::path_ref_match(completion, path_ctx, &ty, &mut item);

    if let Some(import_to_add) = ctx.import_to_add {
        item.add_import(import_to_add);
    }
    Some(item)
}

#[derive(Clone, Copy)]
enum Variant {
    Struct(hir::Struct),
    EnumVariant(hir::Variant),
}

impl Variant {
    fn fields(self, ctx: &CompletionContext<'_>) -> Option<Vec<hir::Field>> {
        let fields = match self {
            Variant::Struct(it) => it.fields(ctx.db),
            Variant::EnumVariant(it) => it.fields(ctx.db),
        };
        let (visible_fields, fields_omitted) = match self {
            Variant::Struct(it) => visible_fields(ctx, &fields, it)?,
            Variant::EnumVariant(it) => visible_fields(ctx, &fields, it)?,
        };
        if !fields_omitted {
            Some(visible_fields)
        } else {
            None
        }
    }

    fn kind(self, db: &dyn HirDatabase) -> StructKind {
        match self {
            Variant::Struct(it) => it.kind(db),
            Variant::EnumVariant(it) => it.kind(db),
        }
    }

    fn symbol_kind(self) -> SymbolKind {
        match self {
            Variant::Struct(_) => SymbolKind::Struct,
            Variant::EnumVariant(_) => SymbolKind::Variant,
        }
    }

    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
        match self {
            Variant::Struct(it) => it.docs(db),
            Variant::EnumVariant(it) => it.docs(db),
        }
    }

    fn is_deprecated(self, ctx: &RenderContext<'_>) -> bool {
        match self {
            Variant::Struct(it) => ctx.is_deprecated(it),
            Variant::EnumVariant(it) => ctx.is_deprecated(it),
        }
    }

    fn ty(self, db: &dyn HirDatabase) -> hir::Type {
        match self {
            Variant::Struct(it) => it.ty(db),
            Variant::EnumVariant(it) => it.parent_enum(db).ty(db),
        }
    }
}
