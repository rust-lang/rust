//! Code common to structs, unions, and enum variants.

use crate::render::RenderContext;
use hir::{db::HirDatabase, HasAttrs, HasVisibility, HirDisplay};
use ide_db::SnippetCap;
use itertools::Itertools;

/// A rendered struct, union, or enum variant, split into fields for actual
/// auto-completion (`literal`, using `field: ()`) and display in the
/// completions menu (`detail`, using `field: type`).
pub(crate) struct RenderedCompound {
    pub literal: String,
    pub detail: String,
}

/// Render a record type (or sub-type) to a `RenderedCompound`. Use `None` for
/// the `name` argument for an anonymous type.
pub(crate) fn render_record(
    db: &dyn HirDatabase,
    snippet_cap: Option<SnippetCap>,
    fields: &[hir::Field],
    name: Option<&str>,
) -> RenderedCompound {
    let fields = fields.iter();

    let (completions, types): (Vec<_>, Vec<_>) = fields
        .enumerate()
        .map(|(idx, field)| {
            (
                if snippet_cap.is_some() {
                    format!("{}: ${{{}:()}}", field.name(db), idx + 1)
                } else {
                    format!("{}: ()", field.name(db))
                },
                format!("{}: {}", field.name(db), field.ty(db).display(db)),
            )
        })
        .unzip();
    RenderedCompound {
        literal: format!("{} {{ {} }}", name.unwrap_or(""), completions.iter().format(", ")),
        detail: format!("{} {{ {} }}", name.unwrap_or(""), types.iter().format(", ")),
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
    let fields = fields.iter();

    let (completions, types): (Vec<_>, Vec<_>) = fields
        .enumerate()
        .map(|(idx, field)| {
            (
                if snippet_cap.is_some() {
                    format!("${{{}:()}}", (idx + 1).to_string())
                } else {
                    "()".to_string()
                },
                field.ty(db).display(db).to_string(),
            )
        })
        .unzip();
    RenderedCompound {
        literal: format!("{}({})", name.unwrap_or(""), completions.iter().format(", ")),
        detail: format!("{}({})", name.unwrap_or(""), types.iter().format(", ")),
    }
}

/// Find all the visible fields in a `HasAttrs`. Returns the list of visible
/// fields, plus a boolean for whether the list is comprehensive (contains no
/// private fields and is not marked `#[non_exhaustive]`).
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
