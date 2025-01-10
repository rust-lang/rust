//! This module provides functionality for querying callable information about a token.

use either::Either;
use hir::{InFile, Semantics, Type};
use parser::T;
use syntax::{
    ast::{self, AstChildren, HasArgList, HasAttrs, HasName},
    match_ast, AstNode, NodeOrToken, SyntaxToken,
};

use crate::RootDatabase;

#[derive(Debug)]
pub struct ActiveParameter {
    pub ty: Type,
    pub src: Option<InFile<Either<ast::SelfParam, ast::Param>>>,
}

impl ActiveParameter {
    /// Returns information about the call argument this token is part of.
    pub fn at_token(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) -> Option<Self> {
        let (signature, active_parameter) = callable_for_token(sema, token)?;

        let idx = active_parameter?;
        let mut params = signature.params();
        if idx >= params.len() {
            cov_mark::hit!(too_many_arguments);
            return None;
        }
        let param = params.swap_remove(idx);
        Some(ActiveParameter { ty: param.ty().clone(), src: sema.source(param) })
    }

    pub fn ident(&self) -> Option<ast::Name> {
        self.src.as_ref().and_then(|param| match param.value.as_ref().right()?.pat()? {
            ast::Pat::IdentPat(ident) => ident.name(),
            _ => None,
        })
    }

    pub fn attrs(&self) -> Option<AstChildren<ast::Attr>> {
        self.src.as_ref().and_then(|param| Some(param.value.as_ref().right()?.attrs()))
    }
}

/// Returns a [`hir::Callable`] this token is a part of and its argument index of said callable.
pub fn callable_for_token(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> Option<(hir::Callable, Option<usize>)> {
    // Find the calling expression and its NameRef
    let parent = token.parent()?;
    let calling_node = parent.ancestors().filter_map(ast::CallableExpr::cast).find(|it| {
        it.arg_list()
            .is_some_and(|it| it.syntax().text_range().contains(token.text_range().start()))
    })?;

    callable_for_node(sema, &calling_node, &token)
}

pub fn callable_for_node(
    sema: &Semantics<'_, RootDatabase>,
    calling_node: &ast::CallableExpr,
    token: &SyntaxToken,
) -> Option<(hir::Callable, Option<usize>)> {
    let callable = match calling_node {
        ast::CallableExpr::Call(call) => sema.resolve_expr_as_callable(&call.expr()?),
        ast::CallableExpr::MethodCall(call) => sema.resolve_method_call_as_callable(call),
    }?;
    let active_param = calling_node.arg_list().map(|arg_list| {
        arg_list
            .syntax()
            .children_with_tokens()
            .filter_map(NodeOrToken::into_token)
            .filter(|t| t.kind() == T![,])
            .take_while(|t| t.text_range().start() <= token.text_range().start())
            .count()
    });
    Some((callable, active_param))
}

pub fn generic_def_for_node(
    sema: &Semantics<'_, RootDatabase>,
    generic_arg_list: &ast::GenericArgList,
    token: &SyntaxToken,
) -> Option<(hir::GenericDef, usize, bool, Option<hir::Variant>)> {
    let parent = generic_arg_list.syntax().parent()?;
    let mut variant = None;
    let def = match_ast! {
        match parent {
            ast::PathSegment(ps) => {
                let res = sema.resolve_path(&ps.parent_path())?;
                let generic_def: hir::GenericDef = match res {
                    hir::PathResolution::Def(hir::ModuleDef::Adt(it)) => it.into(),
                    hir::PathResolution::Def(hir::ModuleDef::Function(it)) => it.into(),
                    hir::PathResolution::Def(hir::ModuleDef::Trait(it)) => it.into(),
                    hir::PathResolution::Def(hir::ModuleDef::TraitAlias(it)) => it.into(),
                    hir::PathResolution::Def(hir::ModuleDef::TypeAlias(it)) => it.into(),
                    hir::PathResolution::Def(hir::ModuleDef::Variant(it)) => {
                        variant = Some(it);
                        it.parent_enum(sema.db).into()
                    },
                    hir::PathResolution::Def(hir::ModuleDef::BuiltinType(_))
                    | hir::PathResolution::Def(hir::ModuleDef::Const(_))
                    | hir::PathResolution::Def(hir::ModuleDef::Macro(_))
                    | hir::PathResolution::Def(hir::ModuleDef::Module(_))
                    | hir::PathResolution::Def(hir::ModuleDef::Static(_)) => return None,
                    hir::PathResolution::BuiltinAttr(_)
                    | hir::PathResolution::ToolModule(_)
                    | hir::PathResolution::Local(_)
                    | hir::PathResolution::TypeParam(_)
                    | hir::PathResolution::ConstParam(_)
                    | hir::PathResolution::SelfType(_)
                    | hir::PathResolution::DeriveHelper(_) => return None,
                };

                generic_def
            },
            ast::AssocTypeArg(_) => {
                // FIXME: We don't record the resolutions for this anywhere atm
                return None;
            },
            ast::MethodCallExpr(mcall) => {
                // recv.method::<$0>()
                let method = sema.resolve_method_call(&mcall)?;
                method.into()
            },
            _ => return None,
        }
    };

    let active_param = generic_arg_list
        .syntax()
        .children_with_tokens()
        .filter_map(NodeOrToken::into_token)
        .filter(|t| t.kind() == T![,])
        .take_while(|t| t.text_range().start() <= token.text_range().start())
        .count();

    let first_arg_is_non_lifetime = generic_arg_list
        .generic_args()
        .next()
        .is_some_and(|arg| !matches!(arg, ast::GenericArg::LifetimeArg(_)));

    Some((def, active_param, first_arg_is_non_lifetime, variant))
}
