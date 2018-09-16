use std::{
    fmt,
    collections::HashMap,
};

use ra_syntax::{
    SyntaxNodeRef, SyntaxNode, SmolStr, AstNode,
    ast::{self, NameOwner, LoopBodyOwner, ArgListOwner},
    algo::{ancestors, generate, walk::preorder}
};

type ScopeId = usize;

#[derive(Debug)]
pub struct FnScopes {
    pub self_param: Option<SyntaxNode>,
    scopes: Vec<ScopeData>,
    scope_for: HashMap<SyntaxNode, ScopeId>,
}

impl FnScopes {
    pub fn new(fn_def: ast::FnDef) -> FnScopes {
        let mut scopes = FnScopes {
            self_param: fn_def.param_list()
                .and_then(|it| it.self_param())
                .map(|it| it.syntax().owned()),
            scopes: Vec::new(),
            scope_for: HashMap::new()
        };
        let root = scopes.root_scope();
        scopes.add_params_bindings(root, fn_def.param_list());
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
    fn add_params_bindings(&mut self, scope: ScopeId, params: Option<ast::ParamList>) {
        params.into_iter()
            .flat_map(|it| it.params())
            .filter_map(|it| it.pat())
            .for_each(|it| self.add_bindings(scope, it));
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

fn compute_block_scopes(block: ast::Block, scopes: &mut FnScopes, mut scope: ScopeId) {
    for stmt in block.statements() {
        match stmt {
            ast::Stmt::LetStmt(stmt) => {
                if let Some(expr) = stmt.initializer() {
                    scopes.set_scope(expr.syntax(), scope);
                    compute_expr_scopes(expr, scopes, scope);
                }
                scope = scopes.new_scope(scope);
                if let Some(pat) = stmt.pat() {
                    scopes.add_bindings(scope, pat);
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
        ast::Expr::BlockExpr(e) => {
            if let Some(block) = e.block() {
                compute_block_scopes(block, scopes, scope);
            }
        }
        ast::Expr::LoopExpr(e) => {
            if let Some(block) = e.loop_body() {
                compute_block_scopes(block, scopes, scope);
            }
        }
        ast::Expr::WhileExpr(e) => {
            let cond_scope = e.condition().and_then(|cond| {
                compute_cond_scopes(cond, scopes, scope)
            });
            if let Some(block) = e.loop_body() {
                compute_block_scopes(block, scopes, cond_scope.unwrap_or(scope));
            }
        }
        ast::Expr::ForExpr(e) => {
            if let Some(expr) = e.iterable() {
                compute_expr_scopes(expr, scopes, scope);
            }
            let mut scope = scope;
            if let Some(pat) = e.pat() {
                scope = scopes.new_scope(scope);
                scopes.add_bindings(scope, pat);
            }
            if let Some(block) = e.loop_body() {
                compute_block_scopes(block, scopes, scope);
            }
        }
        ast::Expr::LambdaExpr(e) => {
            let mut scope = scopes.new_scope(scope);
            scopes.add_params_bindings(scope, e.param_list());
            if let Some(body) = e.body() {
                scopes.set_scope(body.syntax(), scope);
                compute_expr_scopes(body, scopes, scope);
            }
        }
        ast::Expr::CallExpr(e) => {
            compute_call_scopes(e.expr(), e.arg_list(), scopes, scope);
        }
        ast::Expr::MethodCallExpr(e) => {
            compute_call_scopes(e.expr(), e.arg_list(), scopes, scope);
        }
        ast::Expr::MatchExpr(e) => {
            if let Some(expr) = e.expr() {
                compute_expr_scopes(expr, scopes, scope);
            }
            for arm in e.match_arm_list().into_iter().flat_map(|it| it.arms()) {
                let scope = scopes.new_scope(scope);
                for pat in arm.pats() {
                    scopes.add_bindings(scope, pat);
                }
                if let Some(expr) = arm.expr() {
                    compute_expr_scopes(expr, scopes, scope);
                }
            }
        }
        _ => {
            expr.syntax().children()
                .filter_map(ast::Expr::cast)
                .for_each(|expr| compute_expr_scopes(expr, scopes, scope))
        }
    };

    fn compute_call_scopes(
        receiver: Option<ast::Expr>,
        arg_list: Option<ast::ArgList>,
        scopes: &mut FnScopes, scope: ScopeId,
    ) {
        arg_list.into_iter()
            .flat_map(|it| it.args())
            .chain(receiver)
            .for_each(|expr| compute_expr_scopes(expr, scopes, scope));
    }

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

#[derive(Debug)]
struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>
}

#[cfg(test)]
mod tests {
    use super::*;
    use ra_syntax::File;
    use {find_node_at_offset, test_utils::extract_offset};

    fn do_check(code: &str, expected: &[&str]) {
        let (off, code) = extract_offset(code);
        let code = {
            let mut buf = String::new();
            let off = u32::from(off) as usize;
            buf.push_str(&code[..off]);
            buf.push_str("marker");
            buf.push_str(&code[off..]);
            buf
        };
        let file = File::parse(&code);
        let marker: ast::PathExpr = find_node_at_offset(file.syntax(), off).unwrap();
        let fn_def: ast::FnDef = find_node_at_offset(file.syntax(), off).unwrap();
        let scopes = FnScopes::new(fn_def);
        let actual = scopes.scope_chain(marker.syntax())
            .flat_map(|scope| scopes.entries(scope))
            .map(|it| it.name())
            .collect::<Vec<_>>();
        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn test_lambda_scope() {
        do_check(r"
            fn quux(foo: i32) {
                let f = |bar, baz: i32| {
                    <|>
                };
            }",
            &["bar", "baz", "foo"],
        );
    }

    #[test]
    fn test_call_scope() {
        do_check(r"
            fn quux() {
                f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_metod_call_scope() {
        do_check(r"
            fn quux() {
                z.f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_loop_scope() {
        do_check(r"
            fn quux() {
                loop {
                    let x = ();
                    <|>
                };
            }",
            &["x"],
        );
    }

    #[test]
    fn test_match() {
        do_check(r"
            fn quux() {
                match () {
                    Some(x) => {
                        <|>
                    }
                };
            }",
            &["x"],
        );
    }
}
