//! Code common to structs, unions, and enum variants.

use crate::context::CompletionContext;
use hir::{HasAttrs, HasCrate, HasVisibility, HirDisplay, StructKind, sym};
use ide_db::SnippetCap;
use itertools::Itertools;
use syntax::SmolStr;

/// A rendered struct, union, or enum variant, split into fields for actual
/// auto-completion (`literal`, using `field: ()`) and display in the
/// completions menu (`detail`, using `field: type`).
pub(crate) struct RenderedLiteral {
    pub(crate) literal: String,
    pub(crate) detail: String,
}

/// Render a record type (or sub-type) to a `RenderedCompound`. Use `None` for
/// the `name` argument for an anonymous type.
pub(crate) fn render_record_lit(
    ctx: &CompletionContext<'_>,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    path: &str,
) -> RenderedLiteral {
    if snippet_cap.is_none() {
        return RenderedLiteral { literal: path.to_owned(), detail: path.to_owned() };
    }
    let completions = fields.iter().enumerate().format_with(", ", |(idx, field), f| {
        if snippet_cap.is_some() {
            f(&format_args!(
                "{}: ${{{}:()}}",
                field.name(ctx.db).display(ctx.db, ctx.edition),
                idx + 1
            ))
        } else {
            f(&format_args!("{}: ()", field.name(ctx.db).display(ctx.db, ctx.edition)))
        }
    });

    let types = fields.iter().format_with(", ", |field, f| {
        f(&format_args!(
            "{}: {}",
            field.name(ctx.db).display(ctx.db, ctx.edition),
            field.ty(ctx.db).display(ctx.db, ctx.display_target)
        ))
    });

    RenderedLiteral {
        literal: format!("{path} {{ {completions} }}"),
        detail: format!("{path} {{ {types} }}"),
    }
}

/// Render a tuple type (or sub-type) to a `RenderedCompound`. Use `None` for
/// the `name` argument for an anonymous type.
pub(crate) fn render_tuple_lit(
    ctx: &CompletionContext<'_>,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    path: &str,
) -> RenderedLiteral {
    if snippet_cap.is_none() {
        return RenderedLiteral { literal: path.to_owned(), detail: path.to_owned() };
    }
    let completions = fields.iter().enumerate().format_with(", ", |(idx, _), f| {
        if snippet_cap.is_some() {
            f(&format_args!("${{{}:()}}", idx + 1))
        } else {
            f(&format_args!("()"))
        }
    });

    let types = fields
        .iter()
        .format_with(", ", |field, f| f(&field.ty(ctx.db).display(ctx.db, ctx.display_target)));

    RenderedLiteral {
        literal: format!("{path}({completions})"),
        detail: format!("{path}({types})"),
    }
}

/// Find all the visible fields in a given list. Returns the list of visible
/// fields, plus a boolean for whether the list is comprehensive (contains no
/// private fields and its item is not marked `#[non_exhaustive]`).
pub(crate) fn visible_fields(
    ctx: &CompletionContext<'_>,
    fields: &[hir::Field],
    item: impl HasAttrs + HasCrate + Copy,
) -> Option<(Vec<hir::Field>, bool)> {
    let module = ctx.module;
    let n_fields = fields.len();
    let fields = fields
        .iter()
        .filter(|field| field.is_visible_from(ctx.db, module))
        .copied()
        .collect::<Vec<_>>();
    let has_invisible_field = n_fields - fields.len() > 0;
    let is_foreign_non_exhaustive = item.attrs(ctx.db).by_key(sym::non_exhaustive).exists()
        && item.krate(ctx.db) != module.krate();
    let fields_omitted = has_invisible_field || is_foreign_non_exhaustive;
    Some((fields, fields_omitted))
}

/// Format a struct, etc. literal option for display in the completions menu.
pub(crate) fn format_literal_label(
    name: &str,
    kind: StructKind,
    snippet_cap: Option<SnippetCap>,
) -> SmolStr {
    if snippet_cap.is_none() {
        return name.into();
    }
    match kind {
        StructKind::Tuple => SmolStr::from_iter([name, "(…)"]),
        StructKind::Record => SmolStr::from_iter([name, " {…}"]),
        StructKind::Unit => name.into(),
    }
}

/// Format a struct, etc. literal option for lookup used in completions filtering.
pub(crate) fn format_literal_lookup(name: &str, kind: StructKind) -> SmolStr {
    match kind {
        StructKind::Tuple => SmolStr::from_iter([name, "()"]),
        StructKind::Record => SmolStr::from_iter([name, "{}"]),
        StructKind::Unit => name.into(),
    }
}
