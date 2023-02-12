//! This module provides functionality for querying callable information about a token.

use either::Either;
use hir::{Semantics, Type};
use syntax::{
    algo::non_trivia_sibling,
    ast::{self, HasArgList, HasName},
    AstNode, Direction, SyntaxToken, TextRange,
};

use crate::RootDatabase;

#[derive(Debug)]
pub struct ActiveParameter {
    pub ty: Type,
    pub pat: Option<Either<ast::SelfParam, ast::Pat>>,
}

impl ActiveParameter {
    /// Returns information about the call argument this token is part of.
    pub fn at_token(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) -> Option<Self> {
        let (signature, active_parameter) = callable_for_token(sema, token)?;

        let idx = active_parameter?;
        let mut params = signature.params(sema.db);
        if !(idx < params.len()) {
            cov_mark::hit!(too_many_arguments);
            return None;
        }
        let (pat, ty) = params.swap_remove(idx);
        Some(ActiveParameter { ty, pat })
    }

    pub fn ident(&self) -> Option<ast::Name> {
        self.pat.as_ref().and_then(|param| match param {
            Either::Right(ast::Pat::IdentPat(ident)) => ident.name(),
            _ => None,
        })
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
            .map_or(false, |it| it.syntax().text_range().contains(token.text_range().start()))
    })?;

    callable_for_node(sema, &calling_node, &token)
}

pub fn callable_for_node(
    sema: &Semantics<'_, RootDatabase>,
    calling_node: &ast::CallableExpr,
    token: &SyntaxToken,
) -> Option<(hir::Callable, Option<usize>)> {
    let callable = match &calling_node {
        ast::CallableExpr::Call(call) => {
            let expr = call.expr()?;
            sema.type_of_expr(&expr)?.adjusted().as_callable(sema.db)
        }
        ast::CallableExpr::MethodCall(call) => sema.resolve_method_call_as_callable(call),
    }?;
    let active_param = if let Some(arg_list) = calling_node.arg_list() {
        let account_for_ws = |arg: &ast::Expr| {
            let node = arg.syntax().clone();
            let left = non_trivia_sibling(node.clone().into(), Direction::Prev)
                .and_then(|it| it.into_token())?
                .text_range();
            let right = non_trivia_sibling(node.into(), Direction::Next)
                .and_then(|it| it.into_token())?
                .text_range();
            Some(TextRange::new(left.end(), right.start()))
        };
        arg_list
            .args()
            .position(|arg| {
                account_for_ws(&arg)
                    .unwrap_or(arg.syntax().text_range())
                    .contains(token.text_range().start())
            })
            .or(Some(0))
    } else {
        None
    };
    Some((callable, active_param))
}
