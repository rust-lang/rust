//! FIXME: write short doc here

use std::fmt::{self, Display};

use hir::{Docs, Documentation, HasSource, HirDisplay};
use join_to_string::join;
use ra_syntax::ast::{self, AstNode, NameOwner, VisibilityOwner};
use std::convert::From;

use crate::{
    db,
    display::{generic_parameters, where_predicates},
};

#[derive(Debug)]
pub enum CallableKind {
    Function,
    StructConstructor,
    VariantConstructor,
    Macro,
}

/// Contains information about a function signature
#[derive(Debug)]
pub struct FunctionSignature {
    pub kind: CallableKind,
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
        let ast_node = function.source(db).ast;
        FunctionSignature::from(&ast_node).with_doc_opt(doc)
    }

    pub(crate) fn from_struct(db: &db::RootDatabase, st: hir::Struct) -> Option<Self> {
        let node: ast::StructDef = st.source(db).ast;
        match node.kind() {
            ast::StructKind::Named(_) => return None,
            _ => (),
        };

        let params = st
            .fields(db)
            .into_iter()
            .map(|field: hir::StructField| {
                let ty = field.ty(db);
                format!("{}", ty.display(db))
            })
            .collect();

        Some(
            FunctionSignature {
                kind: CallableKind::StructConstructor,
                visibility: node.visibility().map(|n| n.syntax().text().to_string()),
                name: node.name().map(|n| n.text().to_string()),
                ret_type: node.name().map(|n| n.text().to_string()),
                parameters: params,
                generic_parameters: generic_parameters(&node),
                where_predicates: where_predicates(&node),
                doc: None,
            }
            .with_doc_opt(st.docs(db)),
        )
    }

    pub(crate) fn from_enum_variant(
        db: &db::RootDatabase,
        variant: hir::EnumVariant,
    ) -> Option<Self> {
        let node: ast::EnumVariant = variant.source(db).ast;
        match node.kind() {
            ast::StructKind::Named(_) | ast::StructKind::Unit => return None,
            _ => (),
        };

        let parent_name = match variant.parent_enum(db).name(db) {
            Some(name) => name.to_string(),
            None => "missing".into(),
        };

        let name = format!("{}::{}", parent_name, variant.name(db).unwrap());

        let params = variant
            .fields(db)
            .into_iter()
            .map(|field: hir::StructField| {
                let name = field.name(db);
                let ty = field.ty(db);
                format!("{}: {}", name, ty.display(db))
            })
            .collect();

        Some(
            FunctionSignature {
                kind: CallableKind::VariantConstructor,
                visibility: None,
                name: Some(name),
                ret_type: None,
                parameters: params,
                generic_parameters: vec![],
                where_predicates: vec![],
                doc: None,
            }
            .with_doc_opt(variant.docs(db)),
        )
    }

    pub(crate) fn from_macro(db: &db::RootDatabase, macro_def: hir::MacroDef) -> Option<Self> {
        let node: ast::MacroCall = macro_def.source(db).ast;

        let params = vec![];

        Some(
            FunctionSignature {
                kind: CallableKind::Macro,
                visibility: None,
                name: node.name().map(|n| n.text().to_string()),
                ret_type: None,
                parameters: params,
                generic_parameters: vec![],
                where_predicates: vec![],
                doc: None,
            }
            .with_doc_opt(macro_def.docs(db)),
        )
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
            kind: CallableKind::Function,
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
            match self.kind {
                CallableKind::Function => write!(f, "fn {}", name)?,
                CallableKind::StructConstructor => write!(f, "struct {}", name)?,
                CallableKind::VariantConstructor => write!(f, "{}", name)?,
                CallableKind::Macro => write!(f, "{}!", name)?,
            }
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
