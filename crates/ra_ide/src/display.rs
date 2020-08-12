//! This module contains utilities for turning SyntaxNodes and HIR types
//! into types that may be used to render in a UI.

mod navigation_target;
mod short_label;

use syntax::{
    ast::{self, AstNode, AttrsOwner, GenericParamsOwner, NameOwner},
    SyntaxKind::{ATTR, COMMENT},
};

use ast::VisibilityOwner;
use stdx::format_to;

pub use navigation_target::NavigationTarget;
pub(crate) use navigation_target::{ToNav, TryToNav};
pub(crate) use short_label::ShortLabel;

pub(crate) fn function_declaration(node: &ast::Fn) -> String {
    let mut buf = String::new();
    if let Some(vis) = node.visibility() {
        format_to!(buf, "{} ", vis);
    }
    if node.async_token().is_some() {
        format_to!(buf, "async ");
    }
    if node.const_token().is_some() {
        format_to!(buf, "const ");
    }
    if node.unsafe_token().is_some() {
        format_to!(buf, "unsafe ");
    }
    if let Some(abi) = node.abi() {
        // Keyword `extern` is included in the string.
        format_to!(buf, "{} ", abi);
    }
    if let Some(name) = node.name() {
        format_to!(buf, "fn {}", name)
    }
    if let Some(type_params) = node.generic_param_list() {
        format_to!(buf, "{}", type_params);
    }
    if let Some(param_list) = node.param_list() {
        format_to!(buf, "{}", param_list);
    }
    if let Some(ret_type) = node.ret_type() {
        if ret_type.ty().is_some() {
            format_to!(buf, " {}", ret_type);
        }
    }
    if let Some(where_clause) = node.where_clause() {
        format_to!(buf, "\n{}", where_clause);
    }
    buf
}

pub(crate) fn const_label(node: &ast::Const) -> String {
    let label: String = node
        .syntax()
        .children_with_tokens()
        .filter(|child| !(child.kind() == COMMENT || child.kind() == ATTR))
        .map(|node| node.to_string())
        .collect();

    label.trim().to_owned()
}

pub(crate) fn type_label(node: &ast::TypeAlias) -> String {
    let label: String = node
        .syntax()
        .children_with_tokens()
        .filter(|child| !(child.kind() == COMMENT || child.kind() == ATTR))
        .map(|node| node.to_string())
        .collect();

    label.trim().to_owned()
}

pub(crate) fn macro_label(node: &ast::MacroCall) -> String {
    let name = node.name().map(|name| name.syntax().text().to_string()).unwrap_or_default();
    let vis = if node.has_atom_attr("macro_export") { "#[macro_export]\n" } else { "" };
    format!("{}macro_rules! {}", vis, name)
}
