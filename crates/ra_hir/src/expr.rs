//! FIXME: write short doc here

pub(crate) mod lower;
pub(crate) mod scope;
pub(crate) mod validation;

use std::{ops::Index, sync::Arc};

use ra_arena::{map::ArenaMap, Arena};
use ra_syntax::{ast, AstPtr};
use rustc_hash::FxHashMap;

use crate::{db::HirDatabase, DefWithBody, Either, HasSource, Resolver, Source};

pub use self::scope::ExprScopes;

pub use hir_def::expr::{
    ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp, MatchArm,
    Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
};

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    exprs: Arena<ExprId, Expr>,
    pats: Arena<PatId, Pat>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    params: Vec<PatId>,
    /// The `ExprId` of the actual body expression.
    body_expr: ExprId,
}

type ExprPtr = Either<AstPtr<ast::Expr>, AstPtr<ast::RecordField>>;
type ExprSource = Source<ExprPtr>;

type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;
type PatSource = Source<PatPtr>;

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
///
/// One complication here is that, due to macro expansion, a single `Body` might
/// be spread across several files. So, for each ExprId and PatId, we record
/// both the HirFileId and the position inside the file. However, we only store
/// AST -> ExprId mapping for non-macro files, as it is not clear how to handle
/// this properly for macros.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    expr_map: FxHashMap<ExprPtr, ExprId>,
    expr_map_back: ArenaMap<ExprId, ExprSource>,
    pat_map: FxHashMap<PatPtr, PatId>,
    pat_map_back: ArenaMap<PatId, PatSource>,
    field_map: FxHashMap<(ExprId, usize), AstPtr<ast::RecordField>>,
}

impl Body {
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

        let (body, source_map) = lower::lower(db, def.resolver(db), file_id, params, body);
        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    pub fn params(&self) -> &[PatId] {
        &self.params
    }

    pub fn body_expr(&self) -> ExprId {
        self.body_expr
    }

    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter()
    }

    pub fn pats(&self) -> impl Iterator<Item = (PatId, &Pat)> {
        self.pats.iter()
    }
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

impl Index<ExprId> for Body {
    type Output = Expr;

    fn index(&self, expr: ExprId) -> &Expr {
        &self.exprs[expr]
    }
}

impl Index<PatId> for Body {
    type Output = Pat;

    fn index(&self, pat: PatId) -> &Pat {
        &self.pats[pat]
    }
}

impl BodySourceMap {
    pub(crate) fn expr_syntax(&self, expr: ExprId) -> Option<ExprSource> {
        self.expr_map_back.get(expr).copied()
    }

    pub(crate) fn node_expr(&self, node: &ast::Expr) -> Option<ExprId> {
        self.expr_map.get(&Either::A(AstPtr::new(node))).cloned()
    }

    pub(crate) fn pat_syntax(&self, pat: PatId) -> Option<PatSource> {
        self.pat_map_back.get(pat).copied()
    }

    pub(crate) fn node_pat(&self, node: &ast::Pat) -> Option<PatId> {
        self.pat_map.get(&Either::A(AstPtr::new(node))).cloned()
    }

    pub(crate) fn field_syntax(&self, expr: ExprId, field: usize) -> AstPtr<ast::RecordField> {
        self.field_map[&(expr, field)]
    }
}
