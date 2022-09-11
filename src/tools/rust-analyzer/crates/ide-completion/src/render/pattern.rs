//! Renderer for patterns.

use hir::{db::HirDatabase, HasAttrs, Name, StructKind};
use ide_db::SnippetCap;
use itertools::Itertools;
use syntax::SmolStr;

use crate::{
    context::{ParamContext, ParamKind, PathCompletionCtx, PatternContext},
    render::{
        variant::{format_literal_label, format_literal_lookup, visible_fields},
        RenderContext,
    },
    CompletionItem, CompletionItemKind,
};

pub(crate) fn render_struct_pat(
    ctx: RenderContext<'_>,
    pattern_ctx: &PatternContext,
    strukt: hir::Struct,
    local_name: Option<Name>,
) -> Option<CompletionItem> {
    let _p = profile::span("render_struct_pat");

    let fields = strukt.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(ctx.completion, &fields, strukt)?;

    if visible_fields.is_empty() {
        // Matching a struct without matching its fields is pointless, unlike matching a Variant without its fields
        return None;
    }

    let name = local_name.unwrap_or_else(|| strukt.name(ctx.db()));
    let (name, escaped_name) = (name.unescaped().to_smol_str(), name.to_smol_str());
    let kind = strukt.kind(ctx.db());
    let label = format_literal_label(name.as_str(), kind);
    let lookup = format_literal_lookup(name.as_str(), kind);
    let pat = render_pat(&ctx, pattern_ctx, &escaped_name, kind, &visible_fields, fields_omitted)?;

    Some(build_completion(ctx, label, lookup, pat, strukt))
}

pub(crate) fn render_variant_pat(
    ctx: RenderContext<'_>,
    pattern_ctx: &PatternContext,
    path_ctx: Option<&PathCompletionCtx>,
    variant: hir::Variant,
    local_name: Option<Name>,
    path: Option<&hir::ModPath>,
) -> Option<CompletionItem> {
    let _p = profile::span("render_variant_pat");

    let fields = variant.fields(ctx.db());
    let (visible_fields, fields_omitted) = visible_fields(ctx.completion, &fields, variant)?;

    let (name, escaped_name) = match path {
        Some(path) => (path.unescaped().to_string().into(), path.to_string().into()),
        None => {
            let name = local_name.unwrap_or_else(|| variant.name(ctx.db()));
            (name.unescaped().to_smol_str(), name.to_smol_str())
        }
    };

    let (label, lookup, pat) = match path_ctx {
        Some(PathCompletionCtx { has_call_parens: true, .. }) => {
            (name.clone(), name, escaped_name.to_string())
        }
        _ => {
            let kind = variant.kind(ctx.db());
            let label = format_literal_label(name.as_str(), kind);
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

    Some(build_completion(ctx, label, lookup, pat, variant))
}

fn build_completion(
    ctx: RenderContext<'_>,
    label: SmolStr,
    lookup: SmolStr,
    pat: String,
    def: impl HasAttrs + Copy,
) -> CompletionItem {
    let mut item = CompletionItem::new(CompletionItemKind::Binding, ctx.source_range(), label);
    item.set_documentation(ctx.docs(def))
        .set_deprecated(ctx.is_deprecated(def))
        .detail(&pat)
        .lookup_by(lookup)
        .set_relevance(ctx.completion_relevance());
    match ctx.snippet_cap() {
        Some(snippet_cap) => item.insert_snippet(snippet_cap, pat),
        None => item.insert_text(pat),
    };
    item.build()
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
        StructKind::Record => {
            render_record_as_pat(ctx.db(), ctx.snippet_cap(), fields, name, fields_omitted)
        }
        StructKind::Unit => name.to_string(),
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
) -> String {
    let fields = fields.iter();
    match snippet_cap {
        Some(_) => {
            format!(
                "{name} {{ {}{} }}",
                fields.enumerate().format_with(", ", |(idx, field), f| {
                    f(&format_args!("{}${}", field.name(db), idx + 1))
                }),
                if fields_omitted { ", .." } else { "" },
                name = name
            )
        }
        None => {
            format!(
                "{name} {{ {}{} }}",
                fields.map(|field| field.name(db).to_smol_str()).format(", "),
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
