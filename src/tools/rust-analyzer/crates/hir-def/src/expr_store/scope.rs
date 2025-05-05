//! Name resolution for expressions.
use hir_expand::{MacroDefId, name::Name};
use la_arena::{Arena, ArenaMap, Idx, IdxRange, RawIdx};
use triomphe::Arc;

use crate::{
    BlockId, DefWithBodyId,
    db::DefDatabase,
    expr_store::{Body, ExpressionStore, HygieneId},
    hir::{Binding, BindingId, Expr, ExprId, Item, LabelId, Pat, PatId, Statement},
};

pub type ScopeId = Idx<ScopeData>;

#[derive(Debug, PartialEq, Eq)]
pub struct ExprScopes {
    scopes: Arena<ScopeData>,
    scope_entries: Arena<ScopeEntry>,
    scope_by_expr: ArenaMap<ExprId, ScopeId>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeEntry {
    name: Name,
    hygiene: HygieneId,
    binding: BindingId,
}

impl ScopeEntry {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub(crate) fn hygiene(&self) -> HygieneId {
        self.hygiene
    }

    pub fn binding(&self) -> BindingId {
        self.binding
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeData {
    parent: Option<ScopeId>,
    block: Option<BlockId>,
    label: Option<(LabelId, Name)>,
    // FIXME: We can compress this with an enum for this and `label`/`block` if memory usage matters.
    macro_def: Option<Box<MacroDefId>>,
    entries: IdxRange<ScopeEntry>,
}

impl ExprScopes {
    pub(crate) fn expr_scopes_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<ExprScopes> {
        let body = db.body(def);
        let mut scopes = ExprScopes::new_body(&body);
        scopes.shrink_to_fit();
        Arc::new(scopes)
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scope_entries[self.scopes[scope].entries.clone()]
    }

    /// If `scope` refers to a block expression scope, returns the corresponding `BlockId`.
    pub fn block(&self, scope: ScopeId) -> Option<BlockId> {
        self.scopes[scope].block
    }

    /// If `scope` refers to a macro def scope, returns the corresponding `MacroId`.
    #[allow(clippy::borrowed_box)] // If we return `&MacroDefId` we need to move it, this way we just clone the `Box`.
    pub fn macro_def(&self, scope: ScopeId) -> Option<&Box<MacroDefId>> {
        self.scopes[scope].macro_def.as_ref()
    }

    /// If `scope` refers to a labeled expression scope, returns the corresponding `Label`.
    pub fn label(&self, scope: ScopeId) -> Option<(LabelId, Name)> {
        self.scopes[scope].label.clone()
    }

    /// Returns the scopes in ascending order.
    pub fn scope_chain(&self, scope: Option<ScopeId>) -> impl Iterator<Item = ScopeId> + '_ {
        std::iter::successors(scope, move |&scope| self.scopes[scope].parent)
    }

    pub fn resolve_name_in_scope(&self, scope: ScopeId, name: &Name) -> Option<&ScopeEntry> {
        self.scope_chain(Some(scope))
            .find_map(|scope| self.entries(scope).iter().find(|it| it.name == *name))
    }

    pub fn scope_for(&self, expr: ExprId) -> Option<ScopeId> {
        self.scope_by_expr.get(expr).copied()
    }

    pub fn scope_by_expr(&self) -> &ArenaMap<ExprId, ScopeId> {
        &self.scope_by_expr
    }
}

fn empty_entries(idx: usize) -> IdxRange<ScopeEntry> {
    IdxRange::new(Idx::from_raw(RawIdx::from(idx as u32))..Idx::from_raw(RawIdx::from(idx as u32)))
}

impl ExprScopes {
    fn new_body(body: &Body) -> ExprScopes {
        let mut scopes = ExprScopes {
            scopes: Arena::default(),
            scope_entries: Arena::default(),
            scope_by_expr: ArenaMap::with_capacity(body.exprs.len()),
        };
        let mut root = scopes.root_scope();
        if let Some(self_param) = body.self_param {
            scopes.add_bindings(body, root, self_param, body.binding_hygiene(self_param));
        }
        scopes.add_params_bindings(body, root, &body.params);
        compute_expr_scopes(body.body_expr, body, &mut scopes, &mut root);
        scopes
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: None,
            block: None,
            label: None,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label: None,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_labeled_scope(&mut self, parent: ScopeId, label: Option<(LabelId, Name)>) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_block_scope(
        &mut self,
        parent: ScopeId,
        block: Option<BlockId>,
        label: Option<(LabelId, Name)>,
    ) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block,
            label,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_macro_def_scope(&mut self, parent: ScopeId, macro_id: Box<MacroDefId>) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label: None,
            macro_def: Some(macro_id),
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn add_bindings(
        &mut self,
        store: &ExpressionStore,
        scope: ScopeId,
        binding: BindingId,
        hygiene: HygieneId,
    ) {
        let Binding { name, .. } = &store.bindings[binding];
        let entry = self.scope_entries.alloc(ScopeEntry { name: name.clone(), binding, hygiene });
        self.scopes[scope].entries =
            IdxRange::new_inclusive(self.scopes[scope].entries.start()..=entry);
    }

    fn add_pat_bindings(&mut self, store: &ExpressionStore, scope: ScopeId, pat: PatId) {
        let pattern = &store[pat];
        if let Pat::Bind { id, .. } = *pattern {
            self.add_bindings(store, scope, id, store.binding_hygiene(id));
        }

        pattern.walk_child_pats(|pat| self.add_pat_bindings(store, scope, pat));
    }

    fn add_params_bindings(&mut self, store: &ExpressionStore, scope: ScopeId, params: &[PatId]) {
        params.iter().for_each(|pat| self.add_pat_bindings(store, scope, *pat));
    }

    fn set_scope(&mut self, node: ExprId, scope: ScopeId) {
        self.scope_by_expr.insert(node, scope);
    }

    fn shrink_to_fit(&mut self) {
        let ExprScopes { scopes, scope_entries, scope_by_expr } = self;
        scopes.shrink_to_fit();
        scope_entries.shrink_to_fit();
        scope_by_expr.shrink_to_fit();
    }
}

fn compute_block_scopes(
    statements: &[Statement],
    tail: Option<ExprId>,
    store: &ExpressionStore,
    scopes: &mut ExprScopes,
    scope: &mut ScopeId,
) {
    for stmt in statements {
        match stmt {
            Statement::Let { pat, initializer, else_branch, .. } => {
                if let Some(expr) = initializer {
                    compute_expr_scopes(*expr, store, scopes, scope);
                }
                if let Some(expr) = else_branch {
                    compute_expr_scopes(*expr, store, scopes, scope);
                }

                *scope = scopes.new_scope(*scope);
                scopes.add_pat_bindings(store, *scope, *pat);
            }
            Statement::Expr { expr, .. } => {
                compute_expr_scopes(*expr, store, scopes, scope);
            }
            Statement::Item(Item::MacroDef(macro_id)) => {
                *scope = scopes.new_macro_def_scope(*scope, macro_id.clone());
            }
            Statement::Item(Item::Other) => (),
        }
    }
    if let Some(expr) = tail {
        compute_expr_scopes(expr, store, scopes, scope);
    }
}

fn compute_expr_scopes(
    expr: ExprId,
    store: &ExpressionStore,
    scopes: &mut ExprScopes,
    scope: &mut ScopeId,
) {
    let make_label =
        |label: &Option<LabelId>| label.map(|label| (label, store.labels[label].name.clone()));

    let compute_expr_scopes = |scopes: &mut ExprScopes, expr: ExprId, scope: &mut ScopeId| {
        compute_expr_scopes(expr, store, scopes, scope)
    };

    scopes.set_scope(expr, *scope);
    match &store[expr] {
        Expr::Block { statements, tail, id, label } => {
            let mut scope = scopes.new_block_scope(*scope, *id, make_label(label));
            // Overwrite the old scope for the block expr, so that every block scope can be found
            // via the block itself (important for blocks that only contain items, no expressions).
            scopes.set_scope(expr, scope);
            compute_block_scopes(statements, *tail, store, scopes, &mut scope);
        }
        Expr::Const(id) => {
            let mut scope = scopes.root_scope();
            compute_expr_scopes(scopes, *id, &mut scope);
        }
        Expr::Unsafe { id, statements, tail } | Expr::Async { id, statements, tail } => {
            let mut scope = scopes.new_block_scope(*scope, *id, None);
            // Overwrite the old scope for the block expr, so that every block scope can be found
            // via the block itself (important for blocks that only contain items, no expressions).
            scopes.set_scope(expr, scope);
            compute_block_scopes(statements, *tail, store, scopes, &mut scope);
        }
        Expr::Loop { body: body_expr, label } => {
            let mut scope = scopes.new_labeled_scope(*scope, make_label(label));
            compute_expr_scopes(scopes, *body_expr, &mut scope);
        }
        Expr::Closure { args, body: body_expr, .. } => {
            let mut scope = scopes.new_scope(*scope);
            scopes.add_params_bindings(store, scope, args);
            compute_expr_scopes(scopes, *body_expr, &mut scope);
        }
        Expr::Match { expr, arms } => {
            compute_expr_scopes(scopes, *expr, scope);
            for arm in arms.iter() {
                let mut scope = scopes.new_scope(*scope);
                scopes.add_pat_bindings(store, scope, arm.pat);
                if let Some(guard) = arm.guard {
                    scope = scopes.new_scope(scope);
                    compute_expr_scopes(scopes, guard, &mut scope);
                }
                compute_expr_scopes(scopes, arm.expr, &mut scope);
            }
        }
        &Expr::If { condition, then_branch, else_branch } => {
            let mut then_branch_scope = scopes.new_scope(*scope);
            compute_expr_scopes(scopes, condition, &mut then_branch_scope);
            compute_expr_scopes(scopes, then_branch, &mut then_branch_scope);
            if let Some(else_branch) = else_branch {
                compute_expr_scopes(scopes, else_branch, scope);
            }
        }
        &Expr::Let { pat, expr } => {
            compute_expr_scopes(scopes, expr, scope);
            *scope = scopes.new_scope(*scope);
            scopes.add_pat_bindings(store, *scope, pat);
        }
        _ => store.walk_child_exprs(expr, |e| compute_expr_scopes(scopes, e, scope)),
    };
}

#[cfg(test)]
mod tests {
    use base_db::RootQueryDb;
    use hir_expand::{InFile, name::AsName};
    use span::FileId;
    use syntax::{AstNode, algo::find_node_at_offset, ast};
    use test_fixture::WithFixture;
    use test_utils::{assert_eq_text, extract_offset};

    use crate::{
        FunctionId, ModuleDefId, db::DefDatabase, nameres::crate_def_map, test_db::TestDB,
    };

    fn find_function(db: &TestDB, file_id: FileId) -> FunctionId {
        let krate = db.test_crate();
        let crate_def_map = crate_def_map(db, krate);

        let module = crate_def_map.modules_for_file(db, file_id).next().unwrap();
        let (_, def) = crate_def_map[module].scope.entries().next().unwrap();
        match def.take_values().unwrap() {
            ModuleDefId::FunctionId(it) => it,
            _ => panic!(),
        }
    }

    fn do_check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: &[&str]) {
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
        let editioned_file_id = position.file_id;
        let offset = position.offset;

        let (file_id, _) = editioned_file_id.unpack(&db);

        let file_syntax = db.parse(editioned_file_id).syntax_node();
        let marker: ast::PathExpr = find_node_at_offset(&file_syntax, offset).unwrap();
        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_body, source_map) = db.body_with_source_map(function.into());

        let expr_id = source_map
            .node_expr(InFile { file_id: editioned_file_id.into(), value: &marker.into() })
            .unwrap()
            .as_expr()
            .unwrap();
        let scope = scopes.scope_for(expr_id);

        let actual = scopes
            .scope_chain(scope)
            .flat_map(|scope| scopes.entries(scope))
            .map(|it| it.name().as_str())
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

    fn do_check_local_name(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected_offset: u32) {
        let (db, position) = TestDB::with_position(ra_fixture);
        let editioned_file_id = position.file_id;
        let offset = position.offset;

        let (file_id, _) = editioned_file_id.unpack(&db);

        let file = db.parse(editioned_file_id).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), offset).unwrap();

        let function = find_function(&db, file_id);

        let scopes = db.expr_scopes(function.into());
        let (_, source_map) = db.body_with_source_map(function.into());

        let expr_scope = {
            let expr_ast = name_ref.syntax().ancestors().find_map(ast::Expr::cast).unwrap();
            let expr_id = source_map
                .node_expr(InFile { file_id: editioned_file_id.into(), value: &expr_ast })
                .unwrap()
                .as_expr()
                .unwrap();
            scopes.scope_for(expr_id).unwrap()
        };

        let resolved = scopes.resolve_name_in_scope(expr_scope, &name_ref.as_name()).unwrap();
        let pat_src = source_map
            .pat_syntax(*source_map.binding_definitions[&resolved.binding()].first().unwrap())
            .unwrap();

        let local_name = pat_src.value.syntax_node_ptr().to_node(file.syntax());
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
