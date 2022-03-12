//! Code common to structs, unions, and enum variants.

use crate::render::RenderContext;
use hir::{db::HirDatabase, HasAttrs, HasVisibility, HirDisplay, StructKind};
use ide_db::SnippetCap;
use itertools::Itertools;
use syntax::SmolStr;

/// A rendered struct, union, or enum variant, split into fields for actual
/// auto-completion (`literal`, using `field: ()`) and display in the
/// completions menu (`detail`, using `field: type`).
pub(crate) struct RenderedCompound {
    pub(crate) literal: String,
    pub(crate) detail: String,
}

/// Render a record type (or sub-type) to a `RenderedCompound`. Use `None` for
/// the `name` argument for an anonymous type.
pub(crate) fn render_record(
    db: &dyn HirDatabase,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: Option<&str>,
) -> RenderedCompound {
    let completions = fields.iter().enumerate().format_with(", ", |(idx, field), f| {
        if snippet_cap.is_some() {
            f(&format_args!("{}: ${{{}:()}}", field.name(db), idx + 1))
        } else {
            f(&format_args!("{}: ()", field.name(db)))
        }
    });

    let types = fields.iter().format_with(", ", |field, f| {
        f(&format_args!("{}: {}", field.name(db), field.ty(db).display(db)))
    });

    RenderedCompound {
        literal: format!("{} {{ {} }}", name.unwrap_or(""), completions),
        detail: format!("{} {{ {} }}", name.unwrap_or(""), types),
    }
}

/// Render a tuple type (or sub-type) to a `RenderedCompound`. Use `None` for
/// the `name` argument for an anonymous type.
pub(crate) fn render_tuple(
    db: &dyn HirDatabase,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: Option<&str>,
) -> RenderedCompound {
    let completions = fields.iter().enumerate().format_with(", ", |(idx, _), f| {
        if snippet_cap.is_some() {
            f(&format_args!("${{{}:()}}", idx + 1))
        } else {
            f(&format_args!("()"))
        }
    });

    let types = fields.iter().format_with(", ", |field, f| f(&field.ty(db).display(db)));

    RenderedCompound {
        literal: format!("{}({})", name.unwrap_or(""), completions),
        detail: format!("{}({})", name.unwrap_or(""), types),
    }
}

/// Find all the visible fields in a given list. Returns the list of visible
/// fields, plus a boolean for whether the list is comprehensive (contains no
/// private fields and its item is not marked `#[non_exhaustive]`).
pub(crate) fn visible_fields(
    ctx: &RenderContext<'_>,
    fields: &[hir::Field],
    item: impl HasAttrs,
) -> Option<(Vec<hir::Field>, bool)> {
    let module = ctx.completion.module?;
    let n_fields = fields.len();
    let fields = fields
        .iter()
        .filter(|field| field.is_visible_from(ctx.db(), module))
        .copied()
        .collect::<Vec<_>>();

    let fields_omitted =
        n_fields - fields.len() > 0 || item.attrs(ctx.db()).by_key("non_exhaustive").exists();
    Some((fields, fields_omitted))
}

/// Format a struct, etc. literal option for display in the completions menu.
pub(crate) fn format_literal_label(name: &str, kind: StructKind) -> SmolStr {
    match kind {
        StructKind::Tuple => SmolStr::from_iter([name, "(…)"]),
        StructKind::Record => SmolStr::from_iter([name, " {…}"]),
        StructKind::Unit => name.into(),
    }
}
