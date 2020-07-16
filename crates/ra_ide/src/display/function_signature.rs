//! FIXME: write short doc here

// FIXME: this modules relies on strings and AST way too much, and it should be
// rewritten (matklad 2020-05-07)
use std::convert::From;

use hir::Documentation;
use ra_syntax::ast::{self, AstNode, NameOwner, VisibilityOwner};
use stdx::split_delim;

use crate::display::{generic_parameters, where_predicates};

#[derive(Debug)]
pub(crate) enum CallableKind {
    Function,
}

/// Contains information about a function signature
#[derive(Debug)]
pub(crate) struct FunctionSignature {
    pub(crate) kind: CallableKind,
    /// Optional visibility
    pub(crate) visibility: Option<String>,
    /// Qualifiers like `async`, `unsafe`, ...
    pub(crate) qualifier: FunctionQualifier,
    /// Name of the function
    pub(crate) name: Option<String>,
    /// Documentation for the function
    pub(crate) doc: Option<Documentation>,
    /// Generic parameters
    pub(crate) generic_parameters: Vec<String>,
    /// Parameters of the function
    pub(crate) parameters: Vec<String>,
    /// Parameter names of the function
    pub(crate) parameter_names: Vec<String>,
    /// Parameter types of the function
    pub(crate) parameter_types: Vec<String>,
    /// Optional return type
    pub(crate) ret_type: Option<String>,
    /// Where predicates
    pub(crate) where_predicates: Vec<String>,
    /// Self param presence
    pub(crate) has_self_param: bool,
}

#[derive(Debug, Default)]
pub(crate) struct FunctionQualifier {
    // `async` and `const` are mutually exclusive. Do we need to enforcing it here?
    pub(crate) is_async: bool,
    pub(crate) is_const: bool,
    pub(crate) is_unsafe: bool,
    /// The string `extern ".."`
    pub(crate) extern_abi: Option<String>,
}

impl From<&'_ ast::FnDef> for FunctionSignature {
    fn from(node: &ast::FnDef) -> FunctionSignature {
        fn param_list(node: &ast::FnDef) -> (bool, Vec<String>, Vec<String>) {
            let mut res = vec![];
            let mut res_types = vec![];
            let mut has_self_param = false;
            if let Some(param_list) = node.param_list() {
                if let Some(self_param) = param_list.self_param() {
                    has_self_param = true;
                    let raw_param = self_param.syntax().text().to_string();

                    res_types.push(
                        raw_param
                            .split(':')
                            .nth(1)
                            .and_then(|it| it.get(1..))
                            .unwrap_or_else(|| "Self")
                            .to_string(),
                    );
                    res.push(raw_param);
                }

                // macro-generated functions are missing whitespace
                fn fmt_param(param: ast::Param) -> String {
                    let text = param.syntax().text().to_string();
                    match split_delim(&text, ':') {
                        Some((left, right)) => format!("{}: {}", left.trim(), right.trim()),
                        _ => text,
                    }
                }

                res.extend(param_list.params().map(fmt_param));
                res_types.extend(param_list.params().map(|param| {
                    let param_text = param.syntax().text().to_string();
                    match param_text.split(':').nth(1).and_then(|it| it.get(1..)) {
                        Some(it) => it.to_string(),
                        None => param_text,
                    }
                }));
            }
            (has_self_param, res, res_types)
        }

        fn param_name_list(node: &ast::FnDef) -> Vec<String> {
            let mut res = vec![];
            if let Some(param_list) = node.param_list() {
                if let Some(self_param) = param_list.self_param() {
                    res.push(self_param.syntax().text().to_string())
                }

                res.extend(
                    param_list
                        .params()
                        .map(|param| {
                            Some(
                                param
                                    .pat()?
                                    .syntax()
                                    .descendants()
                                    .find_map(ast::Name::cast)?
                                    .text()
                                    .to_string(),
                            )
                        })
                        .map(|param| param.unwrap_or_default()),
                );
            }
            res
        }

        let (has_self_param, parameters, parameter_types) = param_list(node);

        FunctionSignature {
            kind: CallableKind::Function,
            visibility: node.visibility().map(|n| n.syntax().text().to_string()),
            qualifier: FunctionQualifier {
                is_async: node.async_token().is_some(),
                is_const: node.const_token().is_some(),
                is_unsafe: node.unsafe_token().is_some(),
                extern_abi: node.abi().map(|n| n.to_string()),
            },
            name: node.name().map(|n| n.text().to_string()),
            ret_type: node
                .ret_type()
                .and_then(|r| r.type_ref())
                .map(|n| n.syntax().text().to_string()),
            parameters,
            parameter_names: param_name_list(node),
            parameter_types,
            generic_parameters: generic_parameters(node),
            where_predicates: where_predicates(node),
            // docs are processed separately
            doc: None,
            has_self_param,
        }
    }
}
