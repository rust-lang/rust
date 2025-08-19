//! Renderer for patterns.

use hir::{Name, StructKind, db::HirDatabase};
use ide_db::{SnippetCap, documentation::HasDocs};
use itertools::Itertools;
use syntax::{Edition, SmolStr, ToSmolStr};

use crate::{
    CompletionItem, CompletionItemKind,
    context::{ParamContext, ParamKind, PathCompletionCtx, PatternContext},
    render::{
        RenderContext,
        variant::{format_literal_label, format_literal_lookup, visible_fields},
    },
};

pub(crate) fn render_struct_pat(
    ctx: RenderContext<'_>,
    pattern_ctx: &PatternContext,
    strukt: hir::Struct,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let _p = tracing::info_span!("render_struct_pat").entered();

    let fields = strukt.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(ctx.completion, &fields, strukt)?;

    if visible_fields.is_empty() {
        // Matching a struct without matching its fields is pointless, unlike matching a Variant without its fields
        return None;
    }

    let name = local_name.unwrap_or_else(|| strukt.name(ctx.db()));
    let (name, escaped_name) =
        (name.as_str(), name.display(ctx.db(), ctx.completion.edition).to_smolstr());
    let kind = strukt.kind(ctx.db());
    let label = format_literal_label(name, kind, ctx.snippet_cap());
    let lookup = format_literal_lookup(name, kind);
    let pat = render_pat(&ctx, pattern_ctx, &escaped_name, kind, &visible_fields, fields_omitted)?;

    let db = ctx.db();

    Some(build_completion(ctx, label, lookup, pat, strukt, strukt.ty(db), false))
}

pub(crate) fn render_variant_pat(
    ctx: RenderContext<'_>,
    pattern_ctx: &PatternContext,
    path_ctx: Option<&PathCompletionCtx<'_>>,
    variant: hir::Variant,
    local_name: Option<Name>,
    path: Option<&hir::ModPath>,
) -> Option<CompletionItem> {
    let _p = tracing::info_span!("render_variant_pat").entered();

    let fields = variant.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(ctx.completion, &fields, variant)?;
    let enum_ty = variant.parent_enum(ctx.db()).ty(ctx.db());

    let (name, escaped_name) = match path {
        Some(path) => (
            path.display_verbatim(ctx.db()).to_smolstr(),
            path.display(ctx.db(), ctx.completion.edition).to_smolstr(),
        ),
        None => {
            let name = local_name.unwrap_or_else(|| variant.name(ctx.db()));

            (
                name.as_str().to_smolstr(),
                name.display(ctx.db(), ctx.completion.edition).to_smolstr(),
            )
        }
    };

    let (label, lookup, pat) = match path_ctx {
        Some(PathCompletionCtx { has_call_parens: true, .. }) => {
            (name.clone(), name, escaped_name.to_string())
        }
        _ => {
            let kind = variant.kind(ctx.db());
            let label = format_literal_label(name.as_str(), kind, ctx.snippet_cap());
            let lookup = format_literal_lookup(name.as_str(), kind);
            let pat = render_pat(
                &ctx,
                pattern_ctx,
                &escaped_name,
                kind,
                &visible_fields,
                fields_omitted,
            )?;
            (label, lookup, pat)
        }
    };

    Some(build_completion(
        ctx,
        label,
        lookup,
        pat,
        variant,
        enum_ty,
        pattern_ctx.missing_variants.contains(&variant),
    ))
}

fn build_completion(
    ctx: RenderContext<'_>,
    label: SmolStr,
    lookup: SmolStr,
    pat: String,
    def: impl HasDocs + Copy,
    adt_ty: hir::Type<'_>,
    // Missing in context of match statement completions
    is_variant_missing: bool,
) -> CompletionItem {
    let mut relevance = ctx.completion_relevance();

    if is_variant_missing {
        relevance.type_match = super::compute_type_match(ctx.completion, &adt_ty);
    }

    let mut item = CompletionItem::new(
        CompletionItemKind::Binding,
        ctx.source_range(),
        label,
        ctx.completion.edition,
    );
    item.set_documentation(ctx.docs(def))
        .set_deprecated(ctx.is_deprecated(def))
        .detail(&pat)
        .lookup_by(lookup)
        .set_relevance(relevance);
    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, pat),
        None => item.insert_text(pat),
    };
    item.build(ctx.db())
}

fn render_pat(
    ctx: &RenderContext<'_>,
    pattern_ctx: &PatternContext,
    name: &str,
    kind: StructKind,
    fields: &[hir::Field],
    fields_omitted: bool,
) -> Option<String> {
    let mut pat = match kind {
        StructKind::Tuple => render_tuple_as_pat(ctx.snippet_cap(), fields, name, fields_omitted),
        StructKind::Record => render_record_as_pat(
            ctx.db(),
            ctx.snippet_cap(),
            fields,
            name,
            fields_omitted,
            ctx.completion.edition,
        ),
        StructKind::Unit => name.to_owned(),
    };

    let needs_ascription = matches!(
        pattern_ctx,
        PatternContext {
            param_ctx: Some(ParamContext { kind: ParamKind::Function(_), .. }),
            has_type_ascription: false,
            ..
        }
    );
    if needs_ascription {
        pat.push(':');
        pat.push(' ');
        pat.push_str(name);
    }
    if ctx.snippet_cap().is_some() {
        pat.push_str("$0");
    }
    Some(pat)
}

fn render_record_as_pat(
    db: &dyn HirDatabase,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: &str,
    fields_omitted: bool,
    edition: Edition,
) -> String {
    let fields = fields.iter();
    match snippet_cap {
        Some(_) => {
            format!(
                "{name} {{ {}{} }}",
                fields.enumerate().format_with(", ", |(idx, field), f| {
                    f(&format_args!("{}${}", field.name(db).display(db, edition), idx + 1))
                }),
                if fields_omitted { ", .." } else { "" },
                name = name
            )
        }
        None => {
            format!(
                "{name} {{ {}{} }}",
                fields.map(|field| field.name(db).display_no_db(edition).to_smolstr()).format(", "),
                if fields_omitted { ", .." } else { "" },
                name = name
            )
        }
    }
}

fn render_tuple_as_pat(
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: &str,
    fields_omitted: bool,
) -> String {
    let fields = fields.iter();
    match snippet_cap {
        Some(_) => {
            format!(
                "{name}({}{})",
                fields
                    .enumerate()
                    .format_with(", ", |(idx, _), f| { f(&format_args!("${}", idx + 1)) }),
                if fields_omitted { ", .." } else { "" },
                name = name
            )
        }
        None => {
            format!(
                "{name}({}{})",
                fields.enumerate().map(|(idx, _)| idx).format(", "),
                if fields_omitted { ", .." } else { "" },
                name = name
            )
        }
    }
}
