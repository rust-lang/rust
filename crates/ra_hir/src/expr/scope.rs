use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use ra_syntax::{
    AstNode, SyntaxNode, TextUnit, TextRange, SyntaxNodePtr,
    algo::generate,
    ast,
};
use ra_arena::{Arena, RawId, impl_arena_id};

use crate::{
    Name, AsName, Function,
    expr::{PatId, ExprId, Pat, Expr, Body, Statement, BodySyntaxMapping},
    HirDatabase,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(RawId);
impl_arena_id!(ScopeId);

#[derive(Debug, PartialEq, Eq)]
pub struct ExprScopes {
    body: Arc<Body>,
    scopes: Arena<ScopeId, ScopeData>,
    scope_for: FxHashMap<ExprId, ScopeId>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeEntry {
    name: Name,
    pat: PatId,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeData {
    parent: Option<ScopeId>,
    entries: Vec<ScopeEntry>,
}

impl ExprScopes {
    // TODO: This should take something more general than Function
    pub(crate) fn expr_scopes_query(db: &impl HirDatabase, function: Function) -> Arc<ExprScopes> {
        let body = db.body_hir(function);
        let res = ExprScopes::new(body);
        Arc::new(res)
    }

    fn new(body: Arc<Body>) -> ExprScopes {
        let mut scopes = ExprScopes {
            body: body.clone(),
            scopes: Arena::default(),
            scope_for: FxHashMap::default(),
        };
        let root = scopes.root_scope();
        scopes.add_params_bindings(root, body.params());
        compute_expr_scopes(body.body_expr(), &body, &mut scopes, root);
        scopes
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }

    pub fn scope_chain_for<'a>(&'a self, expr: ExprId) -> impl Iterator<Item = ScopeId> + 'a {
        generate(self.scope_for(expr), move |&scope| {
            self.scopes[scope].parent
        })
    }

    pub fn resolve_local_name<'a>(
        &'a self,
        context_expr: ExprId,
        name: Name,
    ) -> Option<&'a ScopeEntry> {
        let mut shadowed = FxHashSet::default();
        let ret = self
            .scope_chain_for(context_expr)
            .flat_map(|scope| self.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .find(|entry| entry.name() == &name);
        ret
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

    fn add_bindings(&mut self, body: &Body, scope: ScopeId, pat: PatId) {
        match &body[pat] {
            Pat::Bind { name, .. } => {
                // bind can have a subpattern, but it's actually not allowed
                // to bind to things in there
                let entry = ScopeEntry {
                    name: name.clone(),
                    pat,
                };
                self.scopes[scope].entries.push(entry)
            }
            p => p.walk_child_pats(|pat| self.add_bindings(body, scope, pat)),
        }
    }

    fn add_params_bindings(&mut self, scope: ScopeId, params: &[PatId]) {
        let body = Arc::clone(&self.body);
        params
            .into_iter()
            .for_each(|pat| self.add_bindings(&body, scope, *pat));
    }

    fn set_scope(&mut self, node: ExprId, scope: ScopeId) {
        self.scope_for.insert(node, scope);
    }

    fn scope_for(&self, expr: ExprId) -> Option<ScopeId> {
        self.scope_for.get(&expr).map(|&scope| scope)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopesWithSyntaxMapping {
    pub syntax_mapping: Arc<BodySyntaxMapping>,
    pub scopes: Arc<ExprScopes>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeEntryWithSyntax {
    name: Name,
    ptr: SyntaxNodePtr,
}

impl ScopeEntryWithSyntax {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn ptr(&self) -> SyntaxNodePtr {
        self.ptr
    }
}

impl ScopesWithSyntaxMapping {
    pub fn scope_chain<'a>(&'a self, node: &SyntaxNode) -> impl Iterator<Item = ScopeId> + 'a {
        generate(self.scope_for(node), move |&scope| {
            self.scopes.scopes[scope].parent
        })
    }

    pub fn scope_chain_for_offset<'a>(
        &'a self,
        offset: TextUnit,
    ) -> impl Iterator<Item = ScopeId> + 'a {
        let scope = self
            .scopes
            .scope_for
            .iter()
            .filter_map(|(id, scope)| Some((self.syntax_mapping.expr_syntax(*id)?, scope)))
            // find containing scope
            .min_by_key(|(ptr, _scope)| {
                (
                    !(ptr.range().start() <= offset && offset <= ptr.range().end()),
                    ptr.range().len(),
                )
            })
            .map(|(ptr, scope)| self.adjust(ptr, *scope, offset));

        generate(scope, move |&scope| self.scopes.scopes[scope].parent)
    }

    // XXX: during completion, cursor might be outside of any particular
    // expression. Try to figure out the correct scope...
    fn adjust(&self, ptr: SyntaxNodePtr, original_scope: ScopeId, offset: TextUnit) -> ScopeId {
        let r = ptr.range();
        let child_scopes = self
            .scopes
            .scope_for
            .iter()
            .filter_map(|(id, scope)| Some((self.syntax_mapping.expr_syntax(*id)?, scope)))
            .map(|(ptr, scope)| (ptr.range(), scope))
            .filter(|(range, _)| range.start() <= offset && range.is_subrange(&r) && *range != r);

        child_scopes
            .max_by(|(r1, _), (r2, _)| {
                if r2.is_subrange(&r1) {
                    std::cmp::Ordering::Greater
                } else if r1.is_subrange(&r2) {
                    std::cmp::Ordering::Less
                } else {
                    r1.start().cmp(&r2.start())
                }
            })
            .map(|(_ptr, scope)| *scope)
            .unwrap_or(original_scope)
    }

    pub fn resolve_local_name(&self, name_ref: &ast::NameRef) -> Option<ScopeEntryWithSyntax> {
        let mut shadowed = FxHashSet::default();
        let name = name_ref.as_name();
        let ret = self
            .scope_chain(name_ref.syntax())
            .flat_map(|scope| self.scopes.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .filter(|entry| entry.name() == &name)
            .nth(0);
        ret.and_then(|entry| {
            Some(ScopeEntryWithSyntax {
                name: entry.name().clone(),
                ptr: self.syntax_mapping.pat_syntax(entry.pat())?,
            })
        })
    }

    pub fn find_all_refs(&self, pat: &ast::BindPat) -> Vec<ReferenceDescriptor> {
        let fn_def = pat.syntax().ancestors().find_map(ast::FnDef::cast).unwrap();
        let name_ptr = SyntaxNodePtr::new(pat.syntax());
        fn_def
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
            .collect()
    }

    fn scope_for(&self, node: &SyntaxNode) -> Option<ScopeId> {
        node.ancestors()
            .map(SyntaxNodePtr::new)
            .filter_map(|ptr| self.syntax_mapping.syntax_expr(ptr))
            .find_map(|it| self.scopes.scope_for(it))
    }
}

impl ScopeEntry {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn pat(&self) -> PatId {
        self.pat
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
            Statement::Let {
                pat, initializer, ..
            } => {
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
        Expr::For {
            iterable,
            pat,
            body: body_expr,
        } => {
            compute_expr_scopes(*iterable, body, scopes, scope);
            let scope = scopes.new_scope(scope);
            scopes.add_bindings(body, scope, *pat);
            compute_expr_scopes(*body_expr, body, scopes, scope);
        }
        Expr::Lambda {
            args,
            body: body_expr,
            ..
        } => {
            let scope = scopes.new_scope(scope);
            scopes.add_params_bindings(scope, &args);
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

#[derive(Debug)]
pub struct ReferenceDescriptor {
    pub range: TextRange,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use ra_syntax::{SourceFile, algo::find_node_at_offset};
    use test_utils::{extract_offset, assert_eq_text};

    use crate::expr;

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
        let file = SourceFile::parse(&code);
        let marker: &ast::PathExpr = find_node_at_offset(file.syntax(), off).unwrap();
        let fn_def: &ast::FnDef = find_node_at_offset(file.syntax(), off).unwrap();
        let body_hir = expr::collect_fn_body_syntax(fn_def);
        let scopes = ExprScopes::new(Arc::clone(body_hir.body()));
        let scopes = ScopesWithSyntaxMapping {
            scopes: Arc::new(scopes),
            syntax_mapping: Arc::new(body_hir),
        };
        let actual = scopes
            .scope_chain(marker.syntax())
            .flat_map(|scope| scopes.scopes.entries(scope))
            .map(|it| it.name().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = expected.join("\n");
        assert_eq_text!(&expected, &actual);
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
    fn test_method_call_scope() {
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
        let file = SourceFile::parse(&code);
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");

        let fn_def: &ast::FnDef = find_node_at_offset(file.syntax(), off).unwrap();
        let name_ref: &ast::NameRef = find_node_at_offset(file.syntax(), off).unwrap();

        let body_hir = expr::collect_fn_body_syntax(fn_def);
        let scopes = ExprScopes::new(Arc::clone(body_hir.body()));
        let scopes = ScopesWithSyntaxMapping {
            scopes: Arc::new(scopes),
            syntax_mapping: Arc::new(body_hir),
        };
        let local_name_entry = scopes.resolve_local_name(name_ref).unwrap();
        let local_name = local_name_entry.ptr();
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
            }
            ",
            53,
        );
    }

    #[test]
    fn ref_patterns_contribute_bindings() {
        do_check_local_name(
            r"
            fn foo() {
                if let Some(&from) = bar() {
                    from<|>;
                }
            }
            ",
            53,
        );
    }
}
