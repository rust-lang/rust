use rustc_hash::{FxHashMap, FxHashSet};

use ra_syntax::{
    AstNode, SmolStr, SyntaxNodeRef, TextRange,
    algo::generate,
    ast::{self, ArgListOwner, LoopBodyOwner, NameOwner},
};

use crate::{
    syntax_ptr::LocalSyntaxPtr,
    arena::{Arena, Id},
};

pub(crate) type ScopeId = Id<ScopeData>;

#[derive(Debug, PartialEq, Eq)]
pub struct FnScopes {
    pub(crate) self_param: Option<LocalSyntaxPtr>,
    scopes: Arena<ScopeData>,
    scope_for: FxHashMap<LocalSyntaxPtr, ScopeId>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeEntry {
    name: SmolStr,
    ptr: LocalSyntaxPtr,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>,
}

impl FnScopes {
    pub(crate) fn new(fn_def: ast::FnDef) -> FnScopes {
        let mut scopes = FnScopes {
            self_param: fn_def
                .param_list()
                .and_then(|it| it.self_param())
                .map(|it| LocalSyntaxPtr::new(it.syntax())),
            scopes: Arena::default(),
            scope_for: FxHashMap::default(),
        };
        let root = scopes.root_scope();
        scopes.add_params_bindings(root, fn_def.param_list());
        if let Some(body) = fn_def.body() {
            compute_block_scopes(body, &mut scopes, root)
        }
        scopes
    }
    pub(crate) fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }
    pub fn scope_chain<'a>(&'a self, node: SyntaxNodeRef) -> impl Iterator<Item = ScopeId> + 'a {
        generate(self.scope_for(node), move |&scope| {
            self.scopes[scope].parent
        })
    }
    pub(crate) fn resolve_local_name<'a>(
        &'a self,
        name_ref: ast::NameRef,
    ) -> Option<&'a ScopeEntry> {
        let mut shadowed = FxHashSet::default();
        let ret = self
            .scope_chain(name_ref.syntax())
            .flat_map(|scope| self.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .filter(|entry| entry.name() == &name_ref.text())
            .nth(0);
        ret
    }

    pub fn find_all_refs(&self, pat: ast::BindPat) -> Vec<ReferenceDescriptor> {
        let fn_def = pat.syntax().ancestors().find_map(ast::FnDef::cast).unwrap();
        let name_ptr = LocalSyntaxPtr::new(pat.syntax());
        let refs: Vec<_> = fn_def
            .syntax()
            .descendants()
            .filter_map(ast::NameRef::cast)
            .filter(|name_ref| match self.resolve_local_name(*name_ref) {
                None => false,
                Some(entry) => entry.ptr() == name_ptr,
            })
            .map(|name_ref| ReferenceDescriptor {
                name: name_ref.syntax().text().to_string(),
                range: name_ref.syntax().range(),
            })
            .collect();

        refs
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: None,
            entries: vec![],
        })
    }
    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            entries: vec![],
        })
    }
    fn add_bindings(&mut self, scope: ScopeId, pat: ast::Pat) {
        let entries = pat
            .syntax()
            .descendants()
            .filter_map(ast::BindPat::cast)
            .filter_map(ScopeEntry::new);
        self.scopes[scope].entries.extend(entries);
    }
    fn add_params_bindings(&mut self, scope: ScopeId, params: Option<ast::ParamList>) {
        params
            .into_iter()
            .flat_map(|it| it.params())
            .filter_map(|it| it.pat())
            .for_each(|it| self.add_bindings(scope, it));
    }
    fn set_scope(&mut self, node: SyntaxNodeRef, scope: ScopeId) {
        self.scope_for.insert(LocalSyntaxPtr::new(node), scope);
    }
    fn scope_for(&self, node: SyntaxNodeRef) -> Option<ScopeId> {
        node.ancestors()
            .map(LocalSyntaxPtr::new)
            .filter_map(|it| self.scope_for.get(&it).map(|&scope| scope))
            .next()
    }
}

impl ScopeEntry {
    fn new(pat: ast::BindPat) -> Option<ScopeEntry> {
        let name = pat.name()?;
        let res = ScopeEntry {
            name: name.text(),
            ptr: LocalSyntaxPtr::new(pat.syntax()),
        };
        Some(res)
    }
    pub(crate) fn name(&self) -> &SmolStr {
        &self.name
    }
    pub(crate) fn ptr(&self) -> LocalSyntaxPtr {
        self.ptr
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
            let cond_scope = e
                .condition()
                .and_then(|cond| compute_cond_scopes(cond, scopes, scope));
            if let Some(block) = e.then_branch() {
                compute_block_scopes(block, scopes, cond_scope.unwrap_or(scope));
            }
            if let Some(block) = e.else_branch() {
                compute_block_scopes(block, scopes, scope);
            }
        }
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
            let cond_scope = e
                .condition()
                .and_then(|cond| compute_cond_scopes(cond, scopes, scope));
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
            let scope = scopes.new_scope(scope);
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
        _ => expr
            .syntax()
            .children()
            .filter_map(ast::Expr::cast)
            .for_each(|expr| compute_expr_scopes(expr, scopes, scope)),
    };

    fn compute_call_scopes(
        receiver: Option<ast::Expr>,
        arg_list: Option<ast::ArgList>,
        scopes: &mut FnScopes,
        scope: ScopeId,
    ) {
        arg_list
            .into_iter()
            .flat_map(|it| it.args())
            .chain(receiver)
            .for_each(|expr| compute_expr_scopes(expr, scopes, scope));
    }

    fn compute_cond_scopes(
        cond: ast::Condition,
        scopes: &mut FnScopes,
        scope: ScopeId,
    ) -> Option<ScopeId> {
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
pub struct ReferenceDescriptor {
    pub range: TextRange,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use ra_editor::find_node_at_offset;
    use ra_syntax::SourceFileNode;
    use test_utils::extract_offset;

    use super::*;

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
        let file = SourceFileNode::parse(&code);
        let marker: ast::PathExpr = find_node_at_offset(file.syntax(), off).unwrap();
        let fn_def: ast::FnDef = find_node_at_offset(file.syntax(), off).unwrap();
        let scopes = FnScopes::new(fn_def);
        let actual = scopes
            .scope_chain(marker.syntax())
            .flat_map(|scope| scopes.entries(scope))
            .map(|it| it.name())
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
    }

    #[test]
    fn test_lambda_scope() {
        do_check(
            r"
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
        do_check(
            r"
            fn quux() {
                f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_metod_call_scope() {
        do_check(
            r"
            fn quux() {
                z.f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_loop_scope() {
        do_check(
            r"
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
        do_check(
            r"
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

    #[test]
    fn test_shadow_variable() {
        do_check(
            r"
            fn foo(x: String) {
                let x : &str = &x<|>;
            }",
            &["x"],
        );
    }

    fn do_check_local_name(code: &str, expected_offset: u32) {
        let (off, code) = extract_offset(code);
        let file = SourceFileNode::parse(&code);
        let fn_def: ast::FnDef = find_node_at_offset(file.syntax(), off).unwrap();
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), off).unwrap();

        let scopes = FnScopes::new(fn_def);

        let local_name_entry = scopes.resolve_local_name(name_ref).unwrap();
        let local_name = local_name_entry.ptr().resolve(&file);
        let expected_name =
            find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into()).unwrap();
        assert_eq!(local_name.range(), expected_name.syntax().range());
    }

    #[test]
    fn test_resolve_local_name() {
        do_check_local_name(
            r#"
            fn foo(x: i32, y: u32) {
                {
                    let z = x * 2;
                }
                {
                    let t = x<|> * 3;
                }
            }"#,
            21,
        );
    }

    #[test]
    fn test_resolve_local_name_declaration() {
        do_check_local_name(
            r#"
            fn foo(x: String) {
                let x : &str = &x<|>;
            }"#,
            21,
        );
    }

    #[test]
    fn test_resolve_local_name_shadow() {
        do_check_local_name(
            r"
        fn foo(x: String) {
            let x : &str = &x;
            x<|>
        }",
            46,
        );
    }
}
