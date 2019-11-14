//! FIXME: write short doc here
use std::sync::Arc;

use hir_expand::name::Name;
use ra_arena::{impl_arena_id, Arena, RawId};
use rustc_hash::FxHashMap;

use crate::{
    body::Body,
    db::DefDatabase2,
    expr::{Expr, ExprId, Pat, PatId, Statement},
    DefWithBodyId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(RawId);
impl_arena_id!(ScopeId);

#[derive(Debug, PartialEq, Eq)]
pub struct ExprScopes {
    scopes: Arena<ScopeId, ScopeData>,
    scope_by_expr: FxHashMap<ExprId, ScopeId>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeEntry {
    name: Name,
    pat: PatId,
}

impl ScopeEntry {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn pat(&self) -> PatId {
        self.pat
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>,
}

impl ExprScopes {
    pub(crate) fn expr_scopes_query(db: &impl DefDatabase2, def: DefWithBodyId) -> Arc<ExprScopes> {
        let body = db.body(def);
        Arc::new(ExprScopes::new(&*body))
    }

    fn new(body: &Body) -> ExprScopes {
        let mut scopes =
            ExprScopes { scopes: Arena::default(), scope_by_expr: FxHashMap::default() };
        let root = scopes.root_scope();
        scopes.add_params_bindings(body, root, body.params());
        compute_expr_scopes(body.body_expr(), body, &mut scopes, root);
        scopes
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }

    pub fn scope_chain(&self, scope: Option<ScopeId>) -> impl Iterator<Item = ScopeId> + '_ {
        std::iter::successors(scope, move |&scope| self.scopes[scope].parent)
    }

    pub fn scope_for(&self, expr: ExprId) -> Option<ScopeId> {
        self.scope_by_expr.get(&expr).copied()
    }

    pub fn scope_by_expr(&self) -> &FxHashMap<ExprId, ScopeId> {
        &self.scope_by_expr
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData { parent: None, entries: vec![] })
    }

    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData { parent: Some(parent), entries: vec![] })
    }

    fn add_bindings(&mut self, body: &Body, scope: ScopeId, pat: PatId) {
        match &body[pat] {
            Pat::Bind { name, .. } => {
                // bind can have a sub pattern, but it's actually not allowed
                // to bind to things in there
                let entry = ScopeEntry { name: name.clone(), pat };
                self.scopes[scope].entries.push(entry)
            }
            p => p.walk_child_pats(|pat| self.add_bindings(body, scope, pat)),
        }
    }

    fn add_params_bindings(&mut self, body: &Body, scope: ScopeId, params: &[PatId]) {
        params.iter().for_each(|pat| self.add_bindings(body, scope, *pat));
    }

    fn set_scope(&mut self, node: ExprId, scope: ScopeId) {
        self.scope_by_expr.insert(node, scope);
    }
}

fn compute_block_scopes(
    statements: &[Statement],
    tail: Option<ExprId>,
    body: &Body,
    scopes: &mut ExprScopes,
    mut scope: ScopeId,
) {
    for stmt in statements {
        match stmt {
            Statement::Let { pat, initializer, .. } => {
                if let Some(expr) = initializer {
                    scopes.set_scope(*expr, scope);
                    compute_expr_scopes(*expr, body, scopes, scope);
                }
                scope = scopes.new_scope(scope);
                scopes.add_bindings(body, scope, *pat);
            }
            Statement::Expr(expr) => {
                scopes.set_scope(*expr, scope);
                compute_expr_scopes(*expr, body, scopes, scope);
            }
        }
    }
    if let Some(expr) = tail {
        compute_expr_scopes(expr, body, scopes, scope);
    }
}

fn compute_expr_scopes(expr: ExprId, body: &Body, scopes: &mut ExprScopes, scope: ScopeId) {
    scopes.set_scope(expr, scope);
    match &body[expr] {
        Expr::Block { statements, tail } => {
            compute_block_scopes(&statements, *tail, body, scopes, scope);
        }
        Expr::For { iterable, pat, body: body_expr } => {
            compute_expr_scopes(*iterable, body, scopes, scope);
            let scope = scopes.new_scope(scope);
            scopes.add_bindings(body, scope, *pat);
            compute_expr_scopes(*body_expr, body, scopes, scope);
        }
        Expr::Lambda { args, body: body_expr, .. } => {
            let scope = scopes.new_scope(scope);
            scopes.add_params_bindings(body, scope, &args);
            compute_expr_scopes(*body_expr, body, scopes, scope);
        }
        Expr::Match { expr, arms } => {
            compute_expr_scopes(*expr, body, scopes, scope);
            for arm in arms {
                let scope = scopes.new_scope(scope);
                for pat in &arm.pats {
                    scopes.add_bindings(body, scope, *pat);
                }
                scopes.set_scope(arm.expr, scope);
                compute_expr_scopes(arm.expr, body, scopes, scope);
            }
        }
        e => e.walk_child_exprs(|e| compute_expr_scopes(e, body, scopes, scope)),
    };
}
