//! Module that contains skip related stuffs.

use rustc_ast::ast;
use rustc_ast_pretty::pprust;
use std::collections::HashSet;

/// Track which blocks of code are to be skipped when formatting.
///
/// You can update it by:
///
/// - attributes slice
/// - manually feeding values into the underlying contexts
///
/// Query this context to know if you need to skip a block.
#[derive(Default, Clone)]
pub(crate) struct SkipContext {
    pub(crate) macros: SkipNameContext,
    pub(crate) attributes: SkipNameContext,
}

impl SkipContext {
    pub(crate) fn update_with_attrs(&mut self, attrs: &[ast::Attribute]) {
        self.macros.extend(get_skip_names("macros", attrs));
        self.attributes.extend(get_skip_names("attributes", attrs));
    }

    pub(crate) fn update(&mut self, other: SkipContext) {
        let SkipContext { macros, attributes } = other;
        self.macros.update(macros);
        self.attributes.update(attributes);
    }
}

/// Track which names to skip.
///
/// Query this context with a string to know whether to skip it.
#[derive(Clone)]
pub(crate) enum SkipNameContext {
    All,
    Values(HashSet<String>),
}

impl Default for SkipNameContext {
    fn default() -> Self {
        Self::Values(Default::default())
    }
}

impl Extend<String> for SkipNameContext {
    fn extend<T: IntoIterator<Item = String>>(&mut self, iter: T) {
        match self {
            Self::All => {}
            Self::Values(values) => values.extend(iter),
        }
    }
}

impl SkipNameContext {
    pub(crate) fn update(&mut self, other: Self) {
        match (self, other) {
            // If we're already skipping everything, nothing more can be added
            (Self::All, _) => {}
            // If we want to skip all, set it
            (this, Self::All) => {
                *this = Self::All;
            }
            // If we have some new values to skip, add them
            (Self::Values(existing_values), Self::Values(new_values)) => {
                existing_values.extend(new_values)
            }
        }
    }

    pub(crate) fn skip(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Values(values) => values.contains(name),
        }
    }

    pub(crate) fn skip_all(&mut self) {
        *self = Self::All;
    }
}

static RUSTFMT: &str = "rustfmt";
static SKIP: &str = "skip";

/// Say if you're playing with `rustfmt`'s skip attribute
pub(crate) fn is_skip_attr(segments: &[ast::PathSegment]) -> bool {
    if segments.len() < 2 || segments[0].ident.to_string() != RUSTFMT {
        return false;
    }
    match segments.len() {
        2 => segments[1].ident.to_string() == SKIP,
        3 => {
            segments[1].ident.to_string() == SKIP
                && ["macros", "attributes"]
                    .iter()
                    .any(|&n| n == pprust::path_segment_to_string(&segments[2]))
        }
        _ => false,
    }
}

fn get_skip_names(kind: &str, attrs: &[ast::Attribute]) -> Vec<String> {
    let mut skip_names = vec![];
    let path = format!("{RUSTFMT}::{SKIP}::{kind}");
    for attr in attrs {
        // rustc_ast::ast::Path is implemented partialEq
        // but it is designed for segments.len() == 1
        if let ast::AttrKind::Normal(normal) = &attr.kind {
            if pprust::path_to_string(&normal.item.path) != path {
                continue;
            }
        }

        if let Some(list) = attr.meta_item_list() {
            for meta_item_inner in list {
                if let Some(name) = meta_item_inner.ident() {
                    skip_names.push(name.to_string());
                }
            }
        }
    }
    skip_names
}
