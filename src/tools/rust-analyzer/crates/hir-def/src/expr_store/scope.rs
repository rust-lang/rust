//! Name resolution for expressions.
use base_db::SourceDatabase;
use hir_expand::{MacroDefId, name::Name};
use la_arena::{Arena, ArenaMap, Idx, IdxRange, RawIdx};

use crate::{
    BlockId, DefWithBodyId, ExpressionStoreOwnerId, GenericDefId, VariantId,
    expr_store::{Body, ExpressionStore, HygieneId, StoreVisitor, StoreVisitorExt, body::Param},
    hir::{
        Binding, BindingId, Expr, ExprId, Item, LabelId, Pat, PatId, Statement,
        generics::GenericParams,
    },
    signatures::VariantFields,
    type_ref::TypeRefId,
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
    kind: ScopeKind,
    entries: IdxRange<ScopeEntry>,
}

#[derive(Debug, PartialEq, Eq)]
enum ScopeKind {
    None,
    Block { id: BlockId, label: Option<LabelId> },
    Label(LabelId),
    MacroDef(Box<MacroDefId>),
}

#[salsa::tracked]
impl ExprScopes {
    #[salsa::tracked(returns(ref))]
    pub fn body_expr_scopes(db: &dyn SourceDatabase, def: DefWithBodyId) -> ExprScopes {
        let body = Body::of(db, def);
        let mut scopes = ExprScopes::new_body(body);
        scopes.shrink_to_fit();
        scopes
    }

    #[salsa::tracked(returns(ref))]
    pub fn sig_expr_scopes(db: &dyn SourceDatabase, def: GenericDefId) -> ExprScopes {
        let (_, store) = GenericParams::with_store(db, def);
        let roots = store.expr_roots();
        let mut scopes = ExprScopes::new_store(store, roots);
        scopes.shrink_to_fit();
        scopes
    }

    #[salsa::tracked(returns(ref))]
    pub fn variant_scopes(db: &dyn SourceDatabase, def: VariantId) -> ExprScopes {
        let fields = VariantFields::of(db, def);
        let roots = fields.store.expr_roots();
        let mut scopes = ExprScopes::new_store(&fields.store, roots);
        scopes.shrink_to_fit();
        scopes
    }
}

impl ExprScopes {
    #[inline]
    pub fn of(db: &dyn SourceDatabase, def: impl Into<ExpressionStoreOwnerId>) -> &ExprScopes {
        match def.into() {
            ExpressionStoreOwnerId::Body(def) => Self::body_expr_scopes(db, def),
            ExpressionStoreOwnerId::Signature(def) => Self::sig_expr_scopes(db, def),
            ExpressionStoreOwnerId::VariantFields(variant_id) => {
                Self::variant_scopes(db, variant_id)
            }
        }
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scope_entries[self.scopes[scope].entries.clone()]
    }

    /// If `scope` refers to a block expression scope, returns the corresponding `BlockId`.
    pub fn block(&self, scope: ScopeId) -> Option<BlockId> {
        match self.scopes[scope].kind {
            ScopeKind::Block { id, label: _ } => Some(id),
            ScopeKind::None | ScopeKind::Label(_) | ScopeKind::MacroDef(_) => None,
        }
    }

    /// If `scope` refers to a macro def scope, returns the corresponding `MacroId`.
    #[allow(clippy::borrowed_box)] // If we return `&MacroDefId` we need to move it, this way we just clone the `Box`.
    pub fn macro_def(&self, scope: ScopeId) -> Option<&Box<MacroDefId>> {
        match &self.scopes[scope].kind {
            ScopeKind::MacroDef(macro_def) => Some(macro_def),
            ScopeKind::None | ScopeKind::Block { id: _, label: _ } | ScopeKind::Label(_) => None,
        }
    }

    /// If `scope` refers to a labeled expression scope, returns the corresponding `Label`.
    pub fn label(&self, scope: ScopeId) -> Option<LabelId> {
        match &self.scopes[scope].kind {
            &ScopeKind::Block { id: _, label } => label,
            &ScopeKind::Label(label) => Some(label),
            ScopeKind::None | ScopeKind::MacroDef(_) => None,
        }
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
            scope_by_expr: ArenaMap::with_capacity(
                body.expr_only.as_ref().map_or(0, |it| it.exprs.len()),
            ),
        };
        let mut root = scopes.root_scope();
        if let Some(Param { formal: self_param, user_written: _ }) = body.self_param {
            scopes.add_bindings(body, root, self_param, body.binding_hygiene(self_param));
        }
        body.params.iter().for_each(|param| scopes.add_pat_bindings(body, root, param.formal));
        ExprScopeVisitor {
            store: body,
            scopes: &mut scopes,
            scope: &mut { root },
            const_scope: &mut root,
        }
        .on_expr(body.root_expr());
        scopes
    }

    fn new_store(store: &ExpressionStore, roots: impl IntoIterator<Item = ExprId>) -> ExprScopes {
        let mut scopes = ExprScopes {
            scopes: Arena::default(),
            scope_entries: Arena::default(),
            scope_by_expr: ArenaMap::with_capacity(
                store.expr_only.as_ref().map_or(0, |it| it.exprs.len()),
            ),
        };
        let root = scopes.root_scope();
        for root_expr in roots {
            let mut scope = scopes.new_scope(root);
            ExprScopeVisitor {
                store,
                scopes: &mut scopes,
                scope: &mut { scope },
                const_scope: &mut scope,
            }
            .on_expr(root_expr);
        }
        scopes
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: None,
            kind: ScopeKind::None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            kind: ScopeKind::None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_labeled_scope(&mut self, parent: ScopeId, label: Option<LabelId>) -> ScopeId {
        let kind = match label {
            Some(label) => ScopeKind::Label(label),
            None => ScopeKind::None,
        };
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            kind,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_block_scope(
        &mut self,
        parent: ScopeId,
        block: Option<BlockId>,
        label: Option<LabelId>,
    ) -> ScopeId {
        let kind = match (block, label) {
            (Some(id), label) => ScopeKind::Block { id, label },
            (None, Some(label)) => ScopeKind::Label(label),
            (None, None) => ScopeKind::None,
        };
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            kind,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_macro_def_scope(&mut self, parent: ScopeId, macro_id: Box<MacroDefId>) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            kind: ScopeKind::MacroDef(macro_id),
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
        let Binding { name, .. } = &store[binding];
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

struct ExprScopeVisitor<'a> {
    store: &'a ExpressionStore,
    scopes: &'a mut ExprScopes,
    scope: &'a mut ScopeId,
    const_scope: &'a mut ScopeId,
}

impl ExprScopeVisitor<'_> {
    fn visit_block(
        &mut self,
        expr: ExprId,
        id: Option<BlockId>,
        statements: &[Statement],
        tail: Option<ExprId>,
        label: Option<LabelId>,
    ) {
        let mut scope = self.scopes.new_block_scope(*self.scope, id, label);
        let mut const_scope = if id.is_some() {
            self.scopes.new_block_scope(*self.const_scope, id, None)
        } else {
            // We don't need to allocate a new scope, since only items matter to us.
            *self.const_scope
        };
        // Overwrite the old scope for the block expr, so that every block scope can be found
        // via the block itself (important for blocks that only contain items, no expressions).
        self.scopes.set_scope(expr, scope);

        let mut visitor = ExprScopeVisitor {
            store: self.store,
            scopes: self.scopes,
            scope: &mut scope,
            const_scope: &mut const_scope,
        };
        for stmt in statements {
            match stmt {
                Statement::Let { pat, initializer, else_branch, type_ref } => {
                    visitor.on_type_opt(*type_ref);
                    visitor.on_expr_opt(*initializer);
                    visitor.on_expr_opt(*else_branch);
                    *visitor.scope = visitor.scopes.new_scope(*visitor.scope);
                    visitor.scopes.add_pat_bindings(visitor.store, *visitor.scope, *pat);
                }
                Statement::Expr { expr, has_semi: _ } => visitor.on_expr(*expr),
                Statement::Item(Item::MacroDef(macro_id)) => {
                    *visitor.scope =
                        visitor.scopes.new_macro_def_scope(*visitor.scope, macro_id.clone());
                    *visitor.const_scope =
                        visitor.scopes.new_macro_def_scope(*visitor.const_scope, macro_id.clone());
                }
                Statement::Item(Item::Other) => (),
            }
        }
        visitor.on_expr_opt(tail);
    }
}

impl StoreVisitor for ExprScopeVisitor<'_> {
    fn on_expr(&mut self, expr: ExprId) {
        self.scopes.set_scope(expr, *self.scope);
        match &self.store[expr] {
            Expr::Block { statements, tail, id, label } => {
                self.visit_block(expr, *id, statements, *tail, *label);
            }
            Expr::Const(expr) => self.on_anon_const_expr(*expr),
            Expr::Unsafe { id, statements, tail } => {
                self.visit_block(expr, *id, statements, *tail, None);
            }
            Expr::Loop { body, label, source: _ } => {
                let mut scope = self.scopes.new_labeled_scope(*self.scope, *label);
                ExprScopeVisitor {
                    store: self.store,
                    scopes: self.scopes,
                    scope: &mut scope,
                    const_scope: self.const_scope,
                }
                .on_expr(*body);
            }
            Expr::Closure { args, arg_types, ret_type, body, capture_by: _, closure_kind: _ } => {
                arg_types.iter().flatten().for_each(|type_ref| self.on_type(*type_ref));
                self.on_type_opt(*ret_type);
                let mut scope = self.scopes.new_scope(*self.scope);
                args.iter().for_each(|arg| self.scopes.add_pat_bindings(self.store, scope, *arg));
                ExprScopeVisitor {
                    store: self.store,
                    scopes: self.scopes,
                    scope: &mut scope,
                    const_scope: self.const_scope,
                }
                .on_expr(*body);
            }
            Expr::Match { expr, arms } => {
                self.on_expr(*expr);
                for arm in arms.iter() {
                    let mut scope = self.scopes.new_scope(*self.scope);
                    self.scopes.add_pat_bindings(self.store, scope, arm.pat);
                    if let Some(guard) = arm.guard {
                        scope = self.scopes.new_scope(scope);
                        ExprScopeVisitor {
                            store: self.store,
                            scopes: self.scopes,
                            scope: &mut scope,
                            const_scope: self.const_scope,
                        }
                        .on_expr(guard);
                    }
                    ExprScopeVisitor {
                        store: self.store,
                        scopes: self.scopes,
                        scope: &mut scope,
                        const_scope: self.const_scope,
                    }
                    .on_expr(arm.expr);
                }
            }
            &Expr::If { condition, then_branch, else_branch } => {
                let mut then_branch_scope = self.scopes.new_scope(*self.scope);
                let mut visitor = ExprScopeVisitor {
                    store: self.store,
                    scopes: self.scopes,
                    scope: &mut then_branch_scope,
                    const_scope: self.const_scope,
                };
                visitor.on_expr(condition);
                visitor.on_expr(then_branch);
                self.on_expr_opt(else_branch);
            }
            &Expr::Let { pat, expr } => {
                self.on_expr(expr);
                *self.scope = self.scopes.new_scope(*self.scope);
                self.scopes.add_pat_bindings(self.store, *self.scope, pat);
            }
            _ => self.store.visit_expr_children(expr, &mut *self),
        }
    }

    fn on_anon_const_expr(&mut self, expr: ExprId) {
        let mut scope = *self.const_scope;
        ExprScopeVisitor {
            store: self.store,
            scopes: self.scopes,
            scope: &mut scope,
            const_scope: self.const_scope,
        }
        .on_expr(expr);
    }

    fn on_pat(&mut self, pat: PatId) {
        self.store.visit_pat_children(pat, &mut *self);
    }

    fn on_type(&mut self, ty: TypeRefId) {
        let mut scope = *self.const_scope;
        self.store.visit_type_ref_children(
            ty,
            ExprScopeVisitor {
                store: self.store,
                scopes: self.scopes,
                scope: &mut scope,
                const_scope: self.const_scope,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use hir_expand::{InFile, name::AsName};
    use span::FileId;
    use syntax::{AstNode, algo::find_node_at_offset, ast};
    use test_fixture::WithFixture;
    use test_utils::{assert_eq_text, extract_offset};

    use crate::{
        DefWithBodyId, FunctionId, ModuleDefId,
        expr_store::{Body, scope::ExprScopes},
        nameres::crate_def_map,
        test_db::TestDB,
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

        let file_syntax = editioned_file_id.parse(&db).syntax_node();
        let marker: Option<ast::PathExpr> = find_node_at_offset(&file_syntax, offset);
        let function = find_function(&db, file_id);

        let scopes = ExprScopes::of(&db, DefWithBodyId::from(function));
        let (body, source_map) = Body::with_source_map(&db, function.into());

        let expr_id = marker
            .and_then(|marker| {
                source_map
                    .node_expr(InFile { file_id: editioned_file_id.into(), value: &marker.into() })
                    .and_then(|expr| expr.as_expr())
            })
            .or_else(|| {
                body.exprs().find_map(|(expr, value)| {
                    let crate::hir::Expr::Path(path) = value else { return None };
                    path.mod_path()
                        .and_then(|path| path.as_ident())
                        .is_some_and(|name| name.as_str() == "marker")
                        .then_some(expr)
                })
            })
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
    fn type_anon_const_scope() {
        do_check(
            r#"
fn f(param: usize) {
    let local = 0;
    let _: [(); $0] = [];
}
"#,
            &["param"],
        );
    }

    #[test]
    fn pattern_type_expr_scope() {
        do_check(
            r#"
fn f(param: usize) {
    let local = 0;
    let _: builtin#pattern_type (usize is 0..=$0) = 0;
}
"#,
            &["param"],
        );
    }

    #[test]
    fn closure_pattern_type_expr_scope() {
        do_check(
            r#"
fn f(param: usize) {
    let local = 0;
    let _ = |_: builtin#pattern_type (usize is 0..=$0)| {};
}
"#,
            &["param"],
        );
    }

    #[test]
    fn array_repeat_expr_scope() {
        do_check(
            r#"
fn f(param: usize) {
    let local = 0;
    let _ = [(); $0];
}
"#,
            &["param"],
        );
    }

    #[test]
    fn inline_const_expr_scope() {
        do_check(
            r#"
fn f(param: usize) {
    let local = 0;
    let _ = const { $0 };
}
"#,
            &["param"],
        );
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

        let file = editioned_file_id.parse(&db).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), offset).unwrap();

        let function = find_function(&db, file_id);

        let scopes = ExprScopes::body_expr_scopes(&db, DefWithBodyId::from(function));
        let (_, source_map) = Body::with_source_map(&db, function.into());

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
        let pat_src =
            source_map.pat_syntax(source_map.patterns_for_binding(resolved.binding())[0]).unwrap();

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
