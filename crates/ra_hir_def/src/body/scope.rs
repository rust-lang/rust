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

    pub fn resolve_name_in_scope(&self, scope: ScopeId, name: &Name) -> Option<&ScopeEntry> {
        self.scope_chain(Some(scope))
            .find_map(|scope| self.entries(scope).iter().find(|it| it.name == *name))
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

#[cfg(test)]
mod tests {
    use hir_expand::{name::AsName, Source};
    use ra_db::{fixture::WithFixture, FileId, SourceDatabase};
    use ra_syntax::{algo::find_node_at_offset, ast, AstNode};
    use test_utils::{assert_eq_text, covers, extract_offset};

    use crate::{db::DefDatabase2, test_db::TestDB, FunctionId, ModuleDefId};

    fn find_function(db: &TestDB, file_id: FileId) -> FunctionId {
        let krate = db.test_crate();
        let crate_def_map = db.crate_def_map(krate);

        let module = crate_def_map.modules_for_file(file_id).next().unwrap();
        let (_, res) = crate_def_map[module].scope.entries().next().unwrap();
        match res.def.take_values().unwrap() {
            ModuleDefId::FunctionId(it) => it,
            _ => panic!(),
        }
    }

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

        let (db, file_id) = TestDB::with_single_file(&code);

        let file_syntax = db.parse(file_id).syntax_node();
        let marker: ast::PathExpr = find_node_at_offset(&file_syntax, off).unwrap();
        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_body, source_map) = db.body_with_source_map(function.into());

        let expr_id = source_map
            .node_expr(Source { file_id: file_id.into(), value: &marker.into() })
            .unwrap();
        let scope = scopes.scope_for(expr_id);

        let actual = scopes
            .scope_chain(scope)
            .flat_map(|scope| scopes.entries(scope))
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

        let (db, file_id) = TestDB::with_single_file(&code);

        let file = db.parse(file_id).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), off).unwrap();

        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_body, source_map) = db.body_with_source_map(function.into());

        let expr_scope = {
            let expr_ast = name_ref.syntax().ancestors().find_map(ast::Expr::cast).unwrap();
            let expr_id =
                source_map.node_expr(Source { file_id: file_id.into(), value: &expr_ast }).unwrap();
            scopes.scope_for(expr_id).unwrap()
        };

        let resolved = scopes.resolve_name_in_scope(expr_scope, &name_ref.as_name()).unwrap();
        let pat_src = source_map.pat_syntax(resolved.pat()).unwrap();

        let local_name = pat_src.value.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
        assert_eq!(local_name.range(), expected_name.syntax().text_range());
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

    #[test]
    fn while_let_desugaring() {
        covers!(infer_resolve_while_let);
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<f32> = None;
    while let Option::Some(spam) = foo {
        spam<|>
    }
}
"#,
            75,
        );
    }
}
