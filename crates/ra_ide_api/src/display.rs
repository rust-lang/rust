//! This module contains utilities for rendering turning things into something
//! that may be used to render in UI.
use super::*;
use std::fmt::{self, Display};
use join_to_string::join;
use ra_syntax::{ast::{self, AstNode, NameOwner, VisibilityOwner, TypeParamsOwner}, SyntaxKind::{ATTR, COMMENT}};
use std::convert::From;
use hir::Docs;

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

/// Contains information about a function signature
#[derive(Debug)]
pub struct FunctionSignature {
    /// Optional visibility
    pub visibility: Option<String>,
    /// Name of the function
    pub name: Option<String>,
    /// Documentation for the function
    pub doc: Option<Documentation>,
    /// Generic parameters
    pub generic_parameters: Vec<String>,
    /// Parameters of the function
    pub parameters: Vec<String>,
    /// Optional return type
    pub ret_type: Option<String>,
    /// Where predicates
    pub where_predicates: Vec<String>,
}

impl FunctionSignature {
    pub(crate) fn with_doc_opt(mut self, doc: Option<Documentation>) -> Self {
        self.doc = doc;
        self
    }

    pub(crate) fn from_hir(db: &db::RootDatabase, function: hir::Function) -> Self {
        let doc = function.docs(db);
        let (_, ast_node) = function.source(db);
        FunctionSignature::from(&*ast_node).with_doc_opt(doc)
    }
}

impl From<&'_ ast::FnDef> for FunctionSignature {
    fn from(node: &ast::FnDef) -> FunctionSignature {
        fn param_list(node: &ast::FnDef) -> Vec<String> {
            let mut res = vec![];
            if let Some(param_list) = node.param_list() {
                if let Some(self_param) = param_list.self_param() {
                    res.push(self_param.syntax().text().to_string())
                }

                res.extend(param_list.params().map(|param| param.syntax().text().to_string()));
            }
            res
        }

        FunctionSignature {
            visibility: node.visibility().map(|n| n.syntax().text().to_string()),
            name: node.name().map(|n| n.text().to_string()),
            ret_type: node
                .ret_type()
                .and_then(|r| r.type_ref())
                .map(|n| n.syntax().text().to_string()),
            parameters: param_list(node),
            generic_parameters: generic_parameters(node),
            where_predicates: where_predicates(node),
            // docs are processed separately
            doc: None,
        }
    }
}

impl Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = &self.visibility {
            write!(f, "{} ", t)?;
        }

        if let Some(name) = &self.name {
            write!(f, "fn {}", name)?;
        }

        if !self.generic_parameters.is_empty() {
            join(self.generic_parameters.iter())
                .separator(", ")
                .surround_with("<", ">")
                .to_fmt(f)?;
        }

        join(self.parameters.iter()).separator(", ").surround_with("(", ")").to_fmt(f)?;

        if let Some(t) = &self.ret_type {
            write!(f, " -> {}", t)?;
        }

        if !self.where_predicates.is_empty() {
            write!(f, "\nwhere ")?;
            join(self.where_predicates.iter()).separator(",\n      ").to_fmt(f)?;
        }

        Ok(())
    }
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
