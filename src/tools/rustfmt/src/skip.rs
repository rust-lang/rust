//! Module that contains skip related stuffs.

use rustc_ast::ast;
use rustc_ast_pretty::pprust;

/// Take care of skip name stack. You can update it by attributes slice or
/// by other context. Query this context to know if you need skip a block.
#[derive(Default, Clone)]
pub(crate) struct SkipContext {
    macros: Vec<String>,
    attributes: Vec<String>,
}

impl SkipContext {
    pub(crate) fn update_with_attrs(&mut self, attrs: &[ast::Attribute]) {
        self.macros.append(&mut get_skip_names("macros", attrs));
        self.attributes
            .append(&mut get_skip_names("attributes", attrs));
    }

    pub(crate) fn update(&mut self, mut other: SkipContext) {
        self.macros.append(&mut other.macros);
        self.attributes.append(&mut other.attributes);
    }

    pub(crate) fn skip_macro(&self, name: &str) -> bool {
        self.macros.iter().any(|n| n == name)
    }

    pub(crate) fn skip_attribute(&self, name: &str) -> bool {
        self.attributes.iter().any(|n| n == name)
    }
}

static RUSTFMT: &'static str = "rustfmt";
static SKIP: &'static str = "skip";

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
                    .any(|&n| n == &pprust::path_segment_to_string(&segments[2]))
        }
        _ => false,
    }
}

fn get_skip_names(kind: &str, attrs: &[ast::Attribute]) -> Vec<String> {
    let mut skip_names = vec![];
    let path = format!("{}::{}::{}", RUSTFMT, SKIP, kind);
    for attr in attrs {
        // rustc_ast::ast::Path is implemented partialEq
        // but it is designed for segments.len() == 1
        if let ast::AttrKind::Normal(attr_item, _) = &attr.kind {
            if pprust::path_to_string(&attr_item.path) != path {
                continue;
            }
        }

        if let Some(list) = attr.meta_item_list() {
            for nested_meta_item in list {
                if let Some(name) = nested_meta_item.ident() {
                    skip_names.push(name.to_string());
                }
            }
        }
    }
    skip_names
}
