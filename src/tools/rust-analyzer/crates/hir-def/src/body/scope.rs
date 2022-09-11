//! Name resolution for expressions.
use std::sync::Arc;

use hir_expand::name::Name;
use la_arena::{Arena, Idx};
use rustc_hash::FxHashMap;

use crate::{
    body::Body,
    db::DefDatabase,
    expr::{Expr, ExprId, LabelId, Pat, PatId, Statement},
    BlockId, DefWithBodyId,
};

pub type ScopeId = Idx<ScopeData>;

#[derive(Debug, PartialEq, Eq)]
pub struct ExprScopes {
    scopes: Arena<ScopeData>,
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
    block: Option<BlockId>,
    label: Option<(LabelId, Name)>,
    entries: Vec<ScopeEntry>,
}

impl ExprScopes {
    pub(crate) fn expr_scopes_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<ExprScopes> {
        let body = db.body(def);
        let mut scopes = ExprScopes::new(&*body);
        scopes.shrink_to_fit();
        Arc::new(scopes)
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scopes[scope].entries
    }

    /// If `scope` refers to a block expression scope, returns the corresponding `BlockId`.
    pub fn block(&self, scope: ScopeId) -> Option<BlockId> {
        self.scopes[scope].block
    }

    /// If `scope` refers to a labeled expression scope, returns the corresponding `Label`.
    pub fn label(&self, scope: ScopeId) -> Option<(LabelId, Name)> {
        self.scopes[scope].label.clone()
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
}

impl ExprScopes {
    fn new(body: &Body) -> ExprScopes {
        let mut scopes =
            ExprScopes { scopes: Arena::default(), scope_by_expr: FxHashMap::default() };
        let mut root = scopes.root_scope();
        scopes.add_params_bindings(body, root, &body.params);
        compute_expr_scopes(body.body_expr, body, &mut scopes, &mut root);
        scopes
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData { parent: None, block: None, label: None, entries: vec![] })
    }

    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label: None,
            entries: vec![],
        })
    }

    fn new_labeled_scope(&mut self, parent: ScopeId, label: Option<(LabelId, Name)>) -> ScopeId {
        self.scopes.alloc(ScopeData { parent: Some(parent), block: None, label, entries: vec![] })
    }

    fn new_block_scope(
        &mut self,
        parent: ScopeId,
        block: BlockId,
        label: Option<(LabelId, Name)>,
    ) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: Some(block),
            label,
            entries: vec![],
        })
    }

    fn add_bindings(&mut self, body: &Body, scope: ScopeId, pat: PatId) {
        let pattern = &body[pat];
        if let Pat::Bind { name, .. } = pattern {
            let entry = ScopeEntry { name: name.clone(), pat };
            self.scopes[scope].entries.push(entry);
        }

        pattern.walk_child_pats(|pat| self.add_bindings(body, scope, pat));
    }

    fn add_params_bindings(&mut self, body: &Body, scope: ScopeId, params: &[PatId]) {
        params.iter().for_each(|pat| self.add_bindings(body, scope, *pat));
    }

    fn set_scope(&mut self, node: ExprId, scope: ScopeId) {
        self.scope_by_expr.insert(node, scope);
    }

    fn shrink_to_fit(&mut self) {
        let ExprScopes { scopes, scope_by_expr } = self;
        scopes.shrink_to_fit();
        scopes.values_mut().for_each(|it| it.entries.shrink_to_fit());
        scope_by_expr.shrink_to_fit();
    }
}

fn compute_block_scopes(
    statements: &[Statement],
    tail: Option<ExprId>,
    body: &Body,
    scopes: &mut ExprScopes,
    scope: &mut ScopeId,
) {
    for stmt in statements {
        match stmt {
            Statement::Let { pat, initializer, else_branch, .. } => {
                if let Some(expr) = initializer {
                    compute_expr_scopes(*expr, body, scopes, scope);
                }
                if let Some(expr) = else_branch {
                    compute_expr_scopes(*expr, body, scopes, scope);
                }

                *scope = scopes.new_scope(*scope);
                scopes.add_bindings(body, *scope, *pat);
            }
            Statement::Expr { expr, .. } => {
                compute_expr_scopes(*expr, body, scopes, scope);
            }
        }
    }
    if let Some(expr) = tail {
        compute_expr_scopes(expr, body, scopes, scope);
    }
}

fn compute_expr_scopes(expr: ExprId, body: &Body, scopes: &mut ExprScopes, scope: &mut ScopeId) {
    let make_label =
        |label: &Option<LabelId>| label.map(|label| (label, body.labels[label].name.clone()));

    scopes.set_scope(expr, *scope);
    match &body[expr] {
        Expr::Block { statements, tail, id, label } => {
            let mut scope = scopes.new_block_scope(*scope, *id, make_label(label));
            // Overwrite the old scope for the block expr, so that every block scope can be found
            // via the block itself (important for blocks that only contain items, no expressions).
            scopes.set_scope(expr, scope);
            compute_block_scopes(statements, *tail, body, scopes, &mut scope);
        }
        Expr::For { iterable, pat, body: body_expr, label } => {
            compute_expr_scopes(*iterable, body, scopes, scope);
            let mut scope = scopes.new_labeled_scope(*scope, make_label(label));
            scopes.add_bindings(body, scope, *pat);
            compute_expr_scopes(*body_expr, body, scopes, &mut scope);
        }
        Expr::While { condition, body: body_expr, label } => {
            let mut scope = scopes.new_labeled_scope(*scope, make_label(label));
            compute_expr_scopes(*condition, body, scopes, &mut scope);
            compute_expr_scopes(*body_expr, body, scopes, &mut scope);
        }
        Expr::Loop { body: body_expr, label } => {
            let mut scope = scopes.new_labeled_scope(*scope, make_label(label));
            compute_expr_scopes(*body_expr, body, scopes, &mut scope);
        }
        Expr::Closure { args, body: body_expr, .. } => {
            let mut scope = scopes.new_scope(*scope);
            scopes.add_params_bindings(body, scope, args);
            compute_expr_scopes(*body_expr, body, scopes, &mut scope);
        }
        Expr::Match { expr, arms } => {
            compute_expr_scopes(*expr, body, scopes, scope);
            for arm in arms.iter() {
                let mut scope = scopes.new_scope(*scope);
                scopes.add_bindings(body, scope, arm.pat);
                if let Some(guard) = arm.guard {
                    scope = scopes.new_scope(scope);
                    compute_expr_scopes(guard, body, scopes, &mut scope);
                }
                compute_expr_scopes(arm.expr, body, scopes, &mut scope);
            }
        }
        &Expr::If { condition, then_branch, else_branch } => {
            let mut then_branch_scope = scopes.new_scope(*scope);
            compute_expr_scopes(condition, body, scopes, &mut then_branch_scope);
            compute_expr_scopes(then_branch, body, scopes, &mut then_branch_scope);
            if let Some(else_branch) = else_branch {
                compute_expr_scopes(else_branch, body, scopes, scope);
            }
        }
        &Expr::Let { pat, expr } => {
            compute_expr_scopes(expr, body, scopes, scope);
            *scope = scopes.new_scope(*scope);
            scopes.add_bindings(body, *scope, pat);
        }
        e => e.walk_child_exprs(|e| compute_expr_scopes(e, body, scopes, scope)),
    };
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, FileId, SourceDatabase};
    use hir_expand::{name::AsName, InFile};
    use syntax::{algo::find_node_at_offset, ast, AstNode};
    use test_utils::{assert_eq_text, extract_offset};

    use crate::{db::DefDatabase, test_db::TestDB, FunctionId, ModuleDefId};

    fn find_function(db: &TestDB, file_id: FileId) -> FunctionId {
        let krate = db.test_crate();
        let crate_def_map = db.crate_def_map(krate);

        let module = crate_def_map.modules_for_file(file_id).next().unwrap();
        let (_, def) = crate_def_map[module].scope.entries().next().unwrap();
        match def.take_values().unwrap() {
            ModuleDefId::FunctionId(it) => it,
            _ => panic!(),
        }
    }

    fn do_check(ra_fixture: &str, expected: &[&str]) {
        let (offset, code) = extract_offset(ra_fixture);
        let code = {
            let mut buf = String::new();
            let off: usize = offset.into();
            buf.push_str(&code[..off]);
            buf.push_str("$0marker");
            buf.push_str(&code[off..]);
            buf
        };

        let (db, position) = TestDB::with_position(&code);
        let file_id = position.file_id;
        let offset = position.offset;

        let file_syntax = db.parse(file_id).syntax_node();
        let marker: ast::PathExpr = find_node_at_offset(&file_syntax, offset).unwrap();
        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_body, source_map) = db.body_with_source_map(function.into());

        let expr_id = source_map
            .node_expr(InFile { file_id: file_id.into(), value: &marker.into() })
            .unwrap();
        let scope = scopes.scope_for(expr_id);

        let actual = scopes
            .scope_chain(scope)
            .flat_map(|scope| scopes.entries(scope))
            .map(|it| it.name().to_smol_str())
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
                    $0
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
                f(|x| $0 );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_method_call_scope() {
        do_check(
            r"
            fn quux() {
                z.f(|x| $0 );
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
                    $0
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
                        $0
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
                let x : &str = &x$0;
            }",
            &["x"],
        );
    }

    #[test]
    fn test_bindings_after_at() {
        do_check(
            r"
fn foo() {
    match Some(()) {
        opt @ Some(unit) => {
            $0
        }
        _ => {}
    }
}
",
            &["opt", "unit"],
        );
    }

    #[test]
    fn macro_inner_item() {
        do_check(
            r"
            macro_rules! mac {
                () => {{
                    fn inner() {}
                    inner();
                }};
            }

            fn foo() {
                mac!();
                $0
            }
        ",
            &[],
        );
    }

    #[test]
    fn broken_inner_item() {
        do_check(
            r"
            fn foo() {
                trait {}
                $0
            }
        ",
            &[],
        );
    }

    fn do_check_local_name(ra_fixture: &str, expected_offset: u32) {
        let (db, position) = TestDB::with_position(ra_fixture);
        let file_id = position.file_id;
        let offset = position.offset;

        let file = db.parse(file_id).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), offset).unwrap();

        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_body, source_map) = db.body_with_source_map(function.into());

        let expr_scope = {
            let expr_ast = name_ref.syntax().ancestors().find_map(ast::Expr::cast).unwrap();
            let expr_id =
                source_map.node_expr(InFile { file_id: file_id.into(), value: &expr_ast }).unwrap();
            scopes.scope_for(expr_id).unwrap()
        };

        let resolved = scopes.resolve_name_in_scope(expr_scope, &name_ref.as_name()).unwrap();
        let pat_src = source_map.pat_syntax(resolved.pat()).unwrap();

        let local_name = pat_src.value.either(
            |it| it.syntax_node_ptr().to_node(file.syntax()),
            |it| it.syntax_node_ptr().to_node(file.syntax()),
        );
        assert_eq!(local_name.text_range(), expected_name.syntax().text_range());
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
        let t = x$0 * 3;
    }
}
"#,
            7,
        );
    }

    #[test]
    fn test_resolve_local_name_declaration() {
        do_check_local_name(
            r#"
fn foo(x: String) {
    let x : &str = &x$0;
}
"#,
            7,
        );
    }

    #[test]
    fn test_resolve_local_name_shadow() {
        do_check_local_name(
            r"
fn foo(x: String) {
    let x : &str = &x;
    x$0
}
",
            28,
        );
    }

    #[test]
    fn ref_patterns_contribute_bindings() {
        do_check_local_name(
            r"
fn foo() {
    if let Some(&from) = bar() {
        from$0;
    }
}
",
            28,
        );
    }

    #[test]
    fn while_let_adds_binding() {
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<f32> = None;
    while let Option::Some(spam) = foo {
        spam$0
    }
}
"#,
            75,
        );
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<f32> = None;
    while (((let Option::Some(_) = foo))) && let Option::Some(spam) = foo {
        spam$0
    }
}
"#,
            107,
        );
    }

    #[test]
    fn match_guard_if_let() {
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<f32> = None;
    match foo {
        _ if let Option::Some(spam) = foo => spam$0,
    }
}
"#,
            93,
        );
    }

    #[test]
    fn let_chains_can_reference_previous_lets() {
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spa$0m > 1 && let Some(spam) = foo && spam > 1 {}
}
"#,
            61,
        );
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spam > 1 && let Some(spam) = foo && sp$0am > 1 {}
}
"#,
            100,
        );
    }
}
