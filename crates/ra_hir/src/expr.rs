//! FIXME: write short doc here

pub(crate) mod scope;
pub(crate) mod validation;

use std::sync::Arc;

use ra_syntax::{ast, AstPtr};

use crate::{db::HirDatabase, DefWithBody, HasSource, Resolver};

pub use self::scope::ExprScopes;

pub use hir_def::{
    body::{Body, BodySourceMap, ExprPtr, ExprSource, PatPtr, PatSource},
    expr::{
        ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp,
        MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
    },
};

pub(crate) fn body_with_source_map_query(
    db: &impl HirDatabase,
    def: DefWithBody,
) -> (Arc<Body>, Arc<BodySourceMap>) {
    let mut params = None;

    let (file_id, body) = match def {
        DefWithBody::Function(f) => {
            let src = f.source(db);
            params = src.ast.param_list();
            (src.file_id, src.ast.body().map(ast::Expr::from))
        }
        DefWithBody::Const(c) => {
            let src = c.source(db);
            (src.file_id, src.ast.body())
        }
        DefWithBody::Static(s) => {
            let src = s.source(db);
            (src.file_id, src.ast.body())
        }
    };
    let resolver = hir_def::body::MacroResolver::new(db, def.module(db).id);
    let (body, source_map) = Body::new(db, resolver, file_id, params, body);
    (Arc::new(body), Arc::new(source_map))
}

pub(crate) fn body_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<Body> {
    db.body_with_source_map(def).0
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    db: &impl HirDatabase,
    owner: DefWithBody,
    expr_id: ExprId,
) -> Resolver {
    let scopes = db.expr_scopes(owner);
    resolver_for_scope(db, owner, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    db: &impl HirDatabase,
    owner: DefWithBody,
    scope_id: Option<scope::ScopeId>,
) -> Resolver {
    let mut r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(Arc::clone(&scopes), scope);
    }
    r
}
