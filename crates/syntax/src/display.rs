//! This module contains utilities for rendering syntax nodes into a string representing their signature.

use crate::ast::{self, HasAttrs, HasGenericParams, HasName};

use ast::HasVisibility;
use stdx::format_to;

pub fn function_declaration(node: &ast::Fn) -> String {
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
        format_to!(buf, "fn {}", name);
    }
    if let Some(type_params) = node.generic_param_list() {
        format_to!(buf, "{}", type_params);
    }
    if let Some(param_list) = node.param_list() {
        let params: Vec<String> = param_list
            .self_param()
            .into_iter()
            .map(|self_param| self_param.to_string())
            .chain(param_list.params().map(|param| param.to_string()))
            .collect();
        // Useful to inline parameters
        format_to!(buf, "({})", params.join(", "));
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

pub fn macro_label(node: &ast::Macro) -> String {
    let name = node.name();
    let mut s = String::new();
    match node {
        ast::Macro::MacroRules(node) => {
            let vis = if node.has_atom_attr("macro_export") { "#[macro_export] " } else { "" };
            format_to!(s, "{}macro_rules!", vis);
        }
        ast::Macro::MacroDef(node) => {
            if let Some(vis) = node.visibility() {
                format_to!(s, "{} ", vis);
            }
            format_to!(s, "macro");
        }
    }
    if let Some(name) = name {
        format_to!(s, " {}", name);
    }
    s
}

pub fn fn_as_proc_macro_label(node: &ast::Fn) -> String {
    let name = node.name();
    let mut s = String::new();
    if let Some(vis) = node.visibility() {
        format_to!(s, "{} ", vis);
    }
    format_to!(s, "macro");
    if let Some(name) = name {
        format_to!(s, " {}", name);
    }
    s
}
