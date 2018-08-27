use std::{
    fmt,
    collections::HashMap,
};

use libsyntax2::{
    SyntaxNodeRef, SyntaxNode, SmolStr, AstNode,
    ast::{self, NameOwner},
    algo::{ancestors, generate, walk::preorder}
};

fn compute_block_scopes(block: ast::Block, scopes: &mut FnScopes, mut scope: ScopeId) {
    for stmt in block.statements() {
        match stmt {
            ast::Stmt::LetStmt(stmt) => {
                scope = scopes.new_scope(scope);
                if let Some(pat) = stmt.pat() {
                    scopes.add_bindings(scope, pat);
                }
                if let Some(expr) = stmt.initializer() {
                    scopes.set_scope(expr.syntax(), scope)
                }
            }
            ast::Stmt::ExprStmt(expr_stmt) => {
                if let Some(expr) = expr_stmt.expr() {
                    scopes.set_scope(expr.syntax(), scope);
                    compute_expr_scopes(expr, scopes, scope);
                }
            }
        }
    }
    if let Some(expr) = block.expr() {
        scopes.set_scope(expr.syntax(), scope);
        compute_expr_scopes(expr, scopes, scope);
    }
}

fn compute_expr_scopes(expr: ast::Expr, scopes: &mut FnScopes, scope: ScopeId) {
    match expr {
        ast::Expr::IfExpr(e) => {
            let cond_scope = e.condition().and_then(|cond| {
                compute_cond_scopes(cond, scopes, scope)
            });
            if let Some(block) = e.then_branch() {
                compute_block_scopes(block, scopes, cond_scope.unwrap_or(scope));
            }
            if let Some(block) = e.else_branch() {
                compute_block_scopes(block, scopes, scope);
            }
        },
        ast::Expr::WhileExpr(e) => {
            let cond_scope = e.condition().and_then(|cond| {
                compute_cond_scopes(cond, scopes, scope)
            });
            if let Some(block) = e.body() {
                compute_block_scopes(block, scopes, cond_scope.unwrap_or(scope));
            }
        },
        ast::Expr::BlockExpr(e) => {
            if let Some(block) = e.block() {
                compute_block_scopes(block, scopes, scope);
            }
        }
        // ForExpr(e) => TODO,
        _ => {
            expr.syntax().children()
                .filter_map(ast::Expr::cast)
                .for_each(|expr| compute_expr_scopes(expr, scopes, scope))
        }
    };

    fn compute_cond_scopes(cond: ast::Condition, scopes: &mut FnScopes, scope: ScopeId) -> Option<ScopeId> {
        if let Some(expr) = cond.expr() {
            compute_expr_scopes(expr, scopes, scope);
        }
        if let Some(pat) = cond.pat() {
            let s = scopes.new_scope(scope);
            scopes.add_bindings(s, pat);
            Some(s)
        } else {
            None
        }
    }
}

type ScopeId = usize;

#[derive(Debug)]
pub struct FnScopes {
    scopes: Vec<ScopeData>,
    scope_for: HashMap<SyntaxNode, ScopeId>,
}

impl FnScopes {
    pub fn new(fn_def: ast::FnDef) -> FnScopes {
        let mut scopes = FnScopes {
            scopes: Vec::new(),
            scope_for: HashMap::new()
        };
        let root = scopes.root_scope();
        fn_def.param_list().into_iter()
            .flat_map(|it| it.params())
            .filter_map(|it| it.pat())
            .for_each(|it| scopes.add_bindings(root, it));

        if let Some(body) = fn_def.body() {
            compute_block_scopes(body, &mut scopes, root)
        }
        scopes
    }
    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }
    pub fn scope_chain<'a>(&'a self, node: SyntaxNodeRef) -> impl Iterator<Item=ScopeId> + 'a {
        generate(self.scope_for(node), move |&scope| self.scopes[scope].parent)
    }
    fn root_scope(&mut self) -> ScopeId {
        let res = self.scopes.len();
        self.scopes.push(ScopeData { parent: None, entries: vec![] });
        res
    }
    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        let res = self.scopes.len();
        self.scopes.push(ScopeData { parent: Some(parent), entries: vec![] });
        res
    }
    fn add_bindings(&mut self, scope: ScopeId, pat: ast::Pat) {
        let entries = preorder(pat.syntax())
            .filter_map(ast::BindPat::cast)
            .filter_map(ScopeEntry::new);
        self.scopes[scope].entries.extend(entries);
    }
    fn set_scope(&mut self, node: SyntaxNodeRef, scope: ScopeId) {
        self.scope_for.insert(node.owned(), scope);
    }
    fn scope_for(&self, node: SyntaxNodeRef) -> Option<ScopeId> {
        ancestors(node)
            .filter_map(|it| self.scope_for.get(&it.owned()).map(|&scope| scope))
            .next()
    }
}

#[derive(Debug)]
struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>
}

pub struct ScopeEntry {
    syntax: SyntaxNode
}

impl ScopeEntry {
    fn new(pat: ast::BindPat) -> Option<ScopeEntry> {
        if pat.name().is_some() {
            Some(ScopeEntry { syntax: pat.syntax().owned() })
        } else {
            None
        }
    }
    pub fn name(&self) -> SmolStr {
        self.ast().name()
            .unwrap()
            .text()
    }
    fn ast(&self) -> ast::BindPat {
        ast::BindPat::cast(self.syntax.borrowed())
            .unwrap()
    }
}

impl fmt::Debug for ScopeEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ScopeEntry")
         .field("name", &self.name())
         .field("syntax", &self.syntax)
         .finish()
    }
}
