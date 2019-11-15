//! FIXME: write short doc here

pub(crate) mod validation;

use std::sync::Arc;

use ra_syntax::AstPtr;

use crate::{db::HirDatabase, DefWithBody, HasBody, Resolver};

pub use hir_def::{
    body::{
        scope::{ExprScopes, ScopeEntry, ScopeId},
        Body, BodySourceMap, ExprPtr, ExprSource, PatPtr, PatSource,
    },
    expr::{
        ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp,
        MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
    },
};

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    db: &impl HirDatabase,
    owner: DefWithBody,
    expr_id: ExprId,
) -> Resolver {
    let scopes = owner.expr_scopes(db);
    resolver_for_scope(db, owner, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    db: &impl HirDatabase,
    owner: DefWithBody,
    scope_id: Option<ScopeId>,
) -> Resolver {
    let mut r = owner.resolver(db);
    let scopes = owner.expr_scopes(db);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(owner, Arc::clone(&scopes), scope);
    }
    r
}
