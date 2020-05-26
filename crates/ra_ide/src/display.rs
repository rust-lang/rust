//! This module contains utilities for turning SyntaxNodes and HIR types
//! into types that may be used to render in a UI.

mod function_signature;
mod navigation_target;
mod structure;
mod short_label;

use std::fmt::Display;

use ra_syntax::{
    ast::{self, AstNode, AttrsOwner, NameOwner, TypeParamsOwner},
    SyntaxKind::{ATTR, COMMENT},
};
use stdx::format_to;

pub use function_signature::FunctionSignature;
pub use navigation_target::NavigationTarget;
pub use structure::{file_structure, StructureNode};

pub(crate) use navigation_target::{ToNav, TryToNav};
pub(crate) use short_label::ShortLabel;

pub(crate) fn function_label(node: &ast::FnDef) -> String {
    FunctionSignature::from(node).to_string()
}

pub(crate) fn const_label(node: &ast::ConstDef) -> String {
    let label: String = node
        .syntax()
        .children_with_tokens()
        .filter(|child| !(child.kind() == COMMENT || child.kind() == ATTR))
        .map(|node| node.to_string())
        .collect();

    label.trim().to_owned()
}

pub(crate) fn type_label(node: &ast::TypeAliasDef) -> String {
    let label: String = node
        .syntax()
        .children_with_tokens()
        .filter(|child| !(child.kind() == COMMENT || child.kind() == ATTR))
        .map(|node| node.to_string())
        .collect();

    label.trim().to_owned()
}

pub(crate) fn generic_parameters<N: TypeParamsOwner>(node: &N) -> Vec<String> {
    let mut res = vec![];
    if let Some(type_params) = node.type_param_list() {
        res.extend(type_params.lifetime_params().map(|p| p.syntax().text().to_string()));
        res.extend(type_params.type_params().map(|p| p.syntax().text().to_string()));
    }
    res
}

pub(crate) fn where_predicates<N: TypeParamsOwner>(node: &N) -> Vec<String> {
    let mut res = vec![];
    if let Some(clause) = node.where_clause() {
        res.extend(clause.predicates().map(|p| p.syntax().text().to_string()));
    }
    res
}

pub(crate) fn macro_label(node: &ast::MacroCall) -> String {
    let name = node.name().map(|name| name.syntax().text().to_string()).unwrap_or_default();
    let vis = if node.has_atom_attr("macro_export") { "#[macro_export]\n" } else { "" };
    format!("{}macro_rules! {}", vis, name)
}

pub(crate) fn rust_code_markup(code: &impl Display) -> String {
    rust_code_markup_with_doc(code, None, None)
}

pub(crate) fn rust_code_markup_with_doc(
    code: &impl Display,
    doc: Option<&str>,
    mod_path: Option<&str>,
) -> String {
    let mut buf = String::new();

    if let Some(mod_path) = mod_path {
        if !mod_path.is_empty() {
            format_to!(buf, "```rust\n{}\n```\n\n", mod_path);
        }
    }
    format_to!(buf, "```rust\n{}\n```", code);

    if let Some(doc) = doc {
        format_to!(buf, "\n___");
        format_to!(buf, "\n\n{}", doc);
    }

    buf
}
