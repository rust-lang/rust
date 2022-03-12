//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::{mem, sync::Arc};

use either::Either;
use hir_expand::{
    ast_id_map::{AstIdMap, FileAstId},
    hygiene::Hygiene,
    name::{name, AsName, Name},
    ExpandError, HirFileId, InFile,
};
use la_arena::Arena;
use profile::Count;
use rustc_hash::FxHashMap;
use syntax::{
    ast::{
        self, ArrayExprKind, AstChildren, HasArgList, HasLoopBody, HasName, LiteralKind,
        SlicePatComponents,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};

use crate::{
    adt::StructKind,
    body::{Body, BodySourceMap, Expander, LabelSource, PatPtr, SyntheticSyntax},
    body::{BodyDiagnostic, ExprSource, PatSource},
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint},
    db::DefDatabase,
    expr::{
        dummy_expr_id, Array, BindingAnnotation, Expr, ExprId, Label, LabelId, Literal, MatchArm,
        Pat, PatId, RecordFieldPat, RecordLitField, Statement,
    },
    intern::Interned,
    item_scope::BuiltinShadowMode,
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, BlockLoc, ModuleDefId, UnresolvedMacro,
};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    hygiene: Hygiene,
    file_id: Option<HirFileId>,
    source_ast_id_map: Option<Arc<AstIdMap>>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx {
            db,
            hygiene: Hygiene::new(db.upcast(), file_id),
            file_id: Some(file_id),
            source_ast_id_map: Some(db.ast_id_map(file_id)),
        }
    }

    pub fn with_hygiene(db: &'a dyn DefDatabase, hygiene: &Hygiene) -> Self {
        LowerCtx { db, hygiene: hygiene.clone(), file_id: None, source_ast_id_map: None }
    }

    pub(crate) fn hygiene(&self) -> &Hygiene {
        &self.hygiene
    }

    pub(crate) fn file_id(&self) -> HirFileId {
        self.file_id.unwrap()
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(ast, self)
    }

    pub(crate) fn ast_id<N: AstNode>(&self, item: &N) -> Option<FileAstId<N>> {
        self.source_ast_id_map.as_ref().map(|ast_id_map| ast_id_map.ast_id(item))
    }
}

pub(super) fn lower(
    db: &dyn DefDatabase,
    expander: Expander,
    params: Option<ast::ParamList>,
    body: Option<ast::Expr>,
) -> (Body, BodySourceMap) {
    ExprCollector {
        db,
        source_map: BodySourceMap::default(),
        body: Body {
            exprs: Arena::default(),
            pats: Arena::default(),
            labels: Arena::default(),
            params: Vec::new(),
            body_expr: dummy_expr_id(),
            block_scopes: Vec::new(),
            _c: Count::new(),
            or_pats: Default::default(),
        },
        expander,
        statements_in_scope: Vec::new(),
        name_to_pat_grouping: Default::default(),
        is_lowering_inside_or_pat: false,
    }
    .collect(params, body)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    expander: Expander,
    body: Body,
    source_map: BodySourceMap,
    statements_in_scope: Vec<Statement>,
    // a poor-mans union-find?
    name_to_pat_grouping: FxHashMap<Name, Vec<PatId>>,
    is_lowering_inside_or_pat: bool,
}

impl ExprCollector<'_> {
    fn collect(
        mut self,
        param_list: Option<ast::ParamList>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        if let Some(param_list) = param_list {
            if let Some(self_param) = param_list.self_param() {
                let ptr = AstPtr::new(&self_param);
                let param_pat = self.alloc_pat(
                    Pat::Bind {
                        name: name![self],
                        mode: BindingAnnotation::new(
                            self_param.mut_token().is_some() && self_param.amp_token().is_none(),
                            false,
                        ),
                        subpat: None,
                    },
                    Either::Right(ptr),
                );
                self.body.params.push(param_pat);
            }

            for pat in param_list.params().filter_map(|param| param.pat()) {
                let param_pat = self.collect_pat(pat);
                self.body.params.push(param_pat);
            }
        };

        self.body.body_expr = self.collect_expr_opt(body);
        (self.body, self.source_map)
    }

    fn ctx(&self) -> LowerCtx<'_> {
        LowerCtx::new(self.db, self.expander.current_file_id)
    }

    fn alloc_expr(&mut self, expr: Expr, ptr: AstPtr<ast::Expr>) -> ExprId {
        let src = self.expander.to_source(ptr);
        let id = self.make_expr(expr, Ok(src.clone()));
        self.source_map.expr_map.insert(src, id);
        id
    }
    // desugared exprs don't have ptr, that's wrong and should be fixed
    // somehow.
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.make_expr(expr, Err(SyntheticSyntax))
    }
    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }
    fn make_expr(&mut self, expr: Expr, src: Result<ExprSource, SyntheticSyntax>) -> ExprId {
        let id = self.body.exprs.alloc(expr);
        self.source_map.expr_map_back.insert(id, src);
        id
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.to_source(ptr);
        let id = self.make_pat(pat, Ok(src.clone()));
        self.source_map.pat_map.insert(src, id);
        id
    }
    fn missing_pat(&mut self) -> PatId {
        self.make_pat(Pat::Missing, Err(SyntheticSyntax))
    }
    fn make_pat(&mut self, pat: Pat, src: Result<PatSource, SyntheticSyntax>) -> PatId {
        let id = self.body.pats.alloc(pat);
        self.source_map.pat_map_back.insert(id, src);
        id
    }

    fn alloc_label(&mut self, label: Label, ptr: AstPtr<ast::Label>) -> LabelId {
        let src = self.expander.to_source(ptr);
        let id = self.make_label(label, src.clone());
        self.source_map.label_map.insert(src, id);
        id
    }
    fn make_label(&mut self, label: Label, src: LabelSource) -> LabelId {
        let id = self.body.labels.alloc(label);
        self.source_map.label_map_back.insert(id, src);
        id
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        self.maybe_collect_expr(expr).unwrap_or_else(|| self.missing_expr())
    }

    /// Returns `None` if and only if the expression is `#[cfg]`d out.
    fn maybe_collect_expr(&mut self, expr: ast::Expr) -> Option<ExprId> {
        let syntax_ptr = AstPtr::new(&expr);
        self.check_cfg(&expr)?;

        Some(match expr {
            ast::Expr::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = self.collect_expr_opt(e.condition());

                self.alloc_expr(Expr::If { condition, then_branch, else_branch }, syntax_ptr)
            }
            ast::Expr::LetExpr(e) => {
                let pat = self.collect_pat_opt(e.pat());
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Let { pat, expr }, syntax_ptr)
            }
            ast::Expr::BlockExpr(e) => match e.modifier() {
                Some(ast::BlockModifier::Try(_)) => {
                    let body = self.collect_block(e);
                    self.alloc_expr(Expr::TryBlock { body }, syntax_ptr)
                }
                Some(ast::BlockModifier::Unsafe(_)) => {
                    let body = self.collect_block(e);
                    self.alloc_expr(Expr::Unsafe { body }, syntax_ptr)
                }
                // FIXME: we need to record these effects somewhere...
                Some(ast::BlockModifier::Label(label)) => {
                    let label = self.collect_label(label);
                    let res = self.collect_block(e);
                    match &mut self.body.exprs[res] {
                        Expr::Block { label: block_label, .. } => {
                            *block_label = Some(label);
                        }
                        _ => unreachable!(),
                    }
                    res
                }
                Some(ast::BlockModifier::Async(_)) => {
                    let body = self.collect_block(e);
                    self.alloc_expr(Expr::Async { body }, syntax_ptr)
                }
                Some(ast::BlockModifier::Const(_)) => {
                    let body = self.collect_block(e);
                    self.alloc_expr(Expr::Const { body }, syntax_ptr)
                }
                None => self.collect_block(e),
            },
            ast::Expr::LoopExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::Loop { body, label }, syntax_ptr)
            }
            ast::Expr::WhileExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_block_opt(e.loop_body());

                let condition = self.collect_expr_opt(e.condition());

                self.alloc_expr(Expr::While { condition, body, label }, syntax_ptr)
            }
            ast::Expr::ForExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let iterable = self.collect_expr_opt(e.iterable());
                let pat = self.collect_pat_opt(e.pat());
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::For { iterable, pat, body, label }, syntax_ptr)
            }
            ast::Expr::CallExpr(e) => {
                let callee = self.collect_expr_opt(e.expr());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().filter_map(|e| self.maybe_collect_expr(e)).collect()
                } else {
                    Box::default()
                };
                self.alloc_expr(Expr::Call { callee, args }, syntax_ptr)
            }
            ast::Expr::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.receiver());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().filter_map(|e| self.maybe_collect_expr(e)).collect()
                } else {
                    Box::default()
                };
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args = e
                    .generic_arg_list()
                    .and_then(|it| GenericArgs::from_ast(&self.ctx(), it))
                    .map(Box::new);
                self.alloc_expr(
                    Expr::MethodCall { receiver, method_name, args, generic_args },
                    syntax_ptr,
                )
            }
            ast::Expr::MatchExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let arms = if let Some(match_arm_list) = e.match_arm_list() {
                    match_arm_list
                        .arms()
                        .filter_map(|arm| {
                            self.check_cfg(&arm).map(|()| MatchArm {
                                pat: self.collect_pat_opt(arm.pat()),
                                expr: self.collect_expr_opt(arm.expr()),
                                guard: arm
                                    .guard()
                                    .map(|guard| self.collect_expr_opt(guard.condition())),
                            })
                        })
                        .collect()
                } else {
                    Box::default()
                };
                self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.expander.parse_path(self.db, path))
                    .map(Expr::Path)
                    .unwrap_or(Expr::Missing);
                self.alloc_expr(path, syntax_ptr)
            }
            ast::Expr::ContinueExpr(e) => self.alloc_expr(
                Expr::Continue { label: e.lifetime().map(|l| Name::new_lifetime(&l)) },
                syntax_ptr,
            ),
            ast::Expr::BreakExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(
                    Expr::Break { expr, label: e.lifetime().map(|l| Name::new_lifetime(&l)) },
                    syntax_ptr,
                )
            }
            ast::Expr::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well
                let src = self.expander.to_source(syntax_ptr);
                self.source_map.expr_map.insert(src, inner);
                inner
            }
            ast::Expr::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Return { expr }, syntax_ptr)
            }
            ast::Expr::YieldExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Yield { expr }, syntax_ptr)
            }
            ast::Expr::RecordExpr(e) => {
                let path =
                    e.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let record_lit = if let Some(nfl) = e.record_expr_field_list() {
                    let fields = nfl
                        .fields()
                        .filter_map(|field| {
                            self.check_cfg(&field)?;

                            let name = field.field_name()?.as_name();

                            let expr = match field.expr() {
                                Some(e) => self.collect_expr(e),
                                None => self.missing_expr(),
                            };
                            let src = self.expander.to_source(AstPtr::new(&field));
                            self.source_map.field_map.insert(src.clone(), expr);
                            self.source_map.field_map_back.insert(expr, src);
                            Some(RecordLitField { name, expr })
                        })
                        .collect();
                    let spread = nfl.spread().map(|s| self.collect_expr(s));
                    Expr::RecordLit { path, fields, spread }
                } else {
                    Expr::RecordLit { path, fields: Box::default(), spread: None }
                };

                self.alloc_expr(record_lit, syntax_ptr)
            }
            ast::Expr::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = match e.field_access() {
                    Some(kind) => kind.as_name(),
                    _ => Name::missing(),
                };
                self.alloc_expr(Expr::Field { expr, name }, syntax_ptr)
            }
            ast::Expr::AwaitExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Await { expr }, syntax_ptr)
            }
            ast::Expr::TryExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Try { expr }, syntax_ptr)
            }
            ast::Expr::CastExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let type_ref = Interned::new(TypeRef::from_ast_opt(&self.ctx(), e.ty()));
                self.alloc_expr(Expr::Cast { expr, type_ref }, syntax_ptr)
            }
            ast::Expr::RefExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let raw_tok = e.raw_token().is_some();
                let mutability = if raw_tok {
                    if e.mut_token().is_some() {
                        Mutability::Mut
                    } else if e.const_token().is_some() {
                        Mutability::Shared
                    } else {
                        unreachable!("parser only remaps to raw_token() if matching mutability token follows")
                    }
                } else {
                    Mutability::from_mutable(e.mut_token().is_some())
                };
                let rawness = Rawness::from_raw(raw_tok);
                self.alloc_expr(Expr::Ref { expr, rawness, mutability }, syntax_ptr)
            }
            ast::Expr::PrefixExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                match e.op_kind() {
                    Some(op) => self.alloc_expr(Expr::UnaryOp { expr, op }, syntax_ptr),
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::ClosureExpr(e) => {
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    for param in pl.params() {
                        let pat = self.collect_pat_opt(param.pat());
                        let type_ref =
                            param.ty().map(|it| Interned::new(TypeRef::from_ast(&self.ctx(), it)));
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let ret_type = e
                    .ret_type()
                    .and_then(|r| r.ty())
                    .map(|it| Interned::new(TypeRef::from_ast(&self.ctx(), it)));
                let body = self.collect_expr_opt(e.body());
                self.alloc_expr(
                    Expr::Lambda { args: args.into(), arg_types: arg_types.into(), ret_type, body },
                    syntax_ptr,
                )
            }
            ast::Expr::BinExpr(e) => {
                let lhs = self.collect_expr_opt(e.lhs());
                let rhs = self.collect_expr_opt(e.rhs());
                let op = e.op_kind();
                self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
            }
            ast::Expr::TupleExpr(e) => {
                let exprs = e.fields().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(Expr::Tuple { exprs }, syntax_ptr)
            }
            ast::Expr::BoxExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Box { expr }, syntax_ptr)
            }

            ast::Expr::ArrayExpr(e) => {
                let kind = e.kind();

                match kind {
                    ArrayExprKind::ElementList(e) => {
                        let exprs = e.map(|expr| self.collect_expr(expr)).collect();
                        self.alloc_expr(Expr::Array(Array::ElementList(exprs)), syntax_ptr)
                    }
                    ArrayExprKind::Repeat { initializer, repeat } => {
                        let initializer = self.collect_expr_opt(initializer);
                        let repeat = self.collect_expr_opt(repeat);
                        self.alloc_expr(
                            Expr::Array(Array::Repeat { initializer, repeat }),
                            syntax_ptr,
                        )
                    }
                }
            }

            ast::Expr::Literal(e) => self.alloc_expr(Expr::Literal(e.kind().into()), syntax_ptr),
            ast::Expr::IndexExpr(e) => {
                let base = self.collect_expr_opt(e.base());
                let index = self.collect_expr_opt(e.index());
                self.alloc_expr(Expr::Index { base, index }, syntax_ptr)
            }
            ast::Expr::RangeExpr(e) => {
                let lhs = e.start().map(|lhs| self.collect_expr(lhs));
                let rhs = e.end().map(|rhs| self.collect_expr(rhs));
                match e.op_kind() {
                    Some(range_type) => {
                        self.alloc_expr(Expr::Range { lhs, rhs, range_type }, syntax_ptr)
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::MacroCall(e) => {
                let macro_ptr = AstPtr::new(&e);
                let mut ids = None;
                self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    ids.get_or_insert(match expansion {
                        Some(it) => this.collect_expr(it),
                        None => this.alloc_expr(Expr::Missing, syntax_ptr.clone()),
                    });
                });
                ids.unwrap_or_else(|| self.alloc_expr(Expr::Missing, syntax_ptr.clone()))
            }
            ast::Expr::MacroStmts(e) => {
                e.statements().for_each(|s| self.collect_stmt(s));
                let tail = e
                    .expr()
                    .map(|e| self.collect_expr(e))
                    .unwrap_or_else(|| self.alloc_expr(Expr::Missing, syntax_ptr.clone()));

                self.alloc_expr(Expr::MacroStmts { tail }, syntax_ptr)
            }
            ast::Expr::UnderscoreExpr(_) => return None,
        })
    }

    fn collect_macro_call<F: FnMut(&mut Self, Option<T>), T: ast::AstNode>(
        &mut self,
        mcall: ast::MacroCall,
        syntax_ptr: AstPtr<ast::MacroCall>,
        record_diagnostics: bool,
        mut collector: F,
    ) {
        // File containing the macro call. Expansion errors will be attached here.
        let outer_file = self.expander.current_file_id;

        let macro_call_ptr = self.expander.to_source(AstPtr::new(&mcall));
        let res = self.expander.enter_expand(self.db, mcall);

        let res = match res {
            Ok(res) => res,
            Err(UnresolvedMacro { path }) => {
                if record_diagnostics {
                    self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedMacroCall {
                        node: InFile::new(outer_file, syntax_ptr),
                        path,
                    });
                }
                collector(self, None);
                return;
            }
        };

        if record_diagnostics {
            match &res.err {
                Some(ExpandError::UnresolvedProcMacro) => {
                    self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedProcMacro {
                        node: InFile::new(outer_file, syntax_ptr),
                    });
                }
                Some(err) => {
                    self.source_map.diagnostics.push(BodyDiagnostic::MacroError {
                        node: InFile::new(outer_file, syntax_ptr),
                        message: err.to_string(),
                    });
                }
                None => {}
            }
        }

        match res.value {
            Some((mark, expansion)) => {
                self.source_map.expansions.insert(macro_call_ptr, self.expander.current_file_id);

                let id = collector(self, Some(expansion));
                self.expander.exit(self.db, mark);
                id
            }
            None => collector(self, None),
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        match expr {
            Some(expr) => self.collect_expr(expr),
            None => self.missing_expr(),
        }
    }

    fn collect_stmt(&mut self, s: ast::Stmt) {
        match s {
            ast::Stmt::LetStmt(stmt) => {
                if self.check_cfg(&stmt).is_none() {
                    return;
                }
                let pat = self.collect_pat_opt(stmt.pat());
                let type_ref =
                    stmt.ty().map(|it| Interned::new(TypeRef::from_ast(&self.ctx(), it)));
                let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                let else_branch = stmt
                    .let_else()
                    .and_then(|let_else| let_else.block_expr())
                    .map(|block| self.collect_block(block));
                self.statements_in_scope.push(Statement::Let {
                    pat,
                    type_ref,
                    initializer,
                    else_branch,
                });
            }
            ast::Stmt::ExprStmt(stmt) => {
                if let Some(expr) = stmt.expr() {
                    if self.check_cfg(&expr).is_none() {
                        return;
                    }
                }
                let has_semi = stmt.semicolon_token().is_some();
                // Note that macro could be expended to multiple statements
                if let Some(ast::Expr::MacroCall(m)) = stmt.expr() {
                    let macro_ptr = AstPtr::new(&m);
                    let syntax_ptr = AstPtr::new(&stmt.expr().unwrap());

                    self.collect_macro_call(
                        m,
                        macro_ptr,
                        false,
                        |this, expansion| match expansion {
                            Some(expansion) => {
                                let statements: ast::MacroStmts = expansion;

                                statements.statements().for_each(|stmt| this.collect_stmt(stmt));
                                if let Some(expr) = statements.expr() {
                                    let expr = this.collect_expr(expr);
                                    this.statements_in_scope
                                        .push(Statement::Expr { expr, has_semi });
                                }
                            }
                            None => {
                                let expr = this.alloc_expr(Expr::Missing, syntax_ptr.clone());
                                this.statements_in_scope.push(Statement::Expr { expr, has_semi });
                            }
                        },
                    );
                } else {
                    let expr = self.collect_expr_opt(stmt.expr());
                    self.statements_in_scope.push(Statement::Expr { expr, has_semi });
                }
            }
            ast::Stmt::Item(_item) => {}
        }
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        let ast_id = self.expander.ast_id(&block);
        let block_loc =
            BlockLoc { ast_id, module: self.expander.def_map.module_id(self.expander.module) };
        let block_id = self.db.intern_block(block_loc);

        let (module, def_map) = match self.db.block_def_map(block_id) {
            Some(def_map) => {
                self.body.block_scopes.push(block_id);
                (def_map.root(), def_map)
            }
            None => (self.expander.module, self.expander.def_map.clone()),
        };
        let prev_def_map = mem::replace(&mut self.expander.def_map, def_map);
        let prev_local_module = mem::replace(&mut self.expander.module, module);
        let prev_statements = std::mem::take(&mut self.statements_in_scope);

        block.statements().for_each(|s| self.collect_stmt(s));
        block.tail_expr().and_then(|e| {
            let expr = self.maybe_collect_expr(e)?;
            self.statements_in_scope.push(Statement::Expr { expr, has_semi: false });
            Some(())
        });

        let mut tail = None;
        if let Some(Statement::Expr { expr, has_semi: false }) = self.statements_in_scope.last() {
            tail = Some(*expr);
            self.statements_in_scope.pop();
        }
        let tail = tail;
        let statements = std::mem::replace(&mut self.statements_in_scope, prev_statements).into();
        let syntax_node_ptr = AstPtr::new(&block.into());
        let expr_id = self.alloc_expr(
            Expr::Block { id: block_id, statements, tail, label: None },
            syntax_node_ptr,
        );

        self.expander.def_map = prev_def_map;
        self.expander.module = prev_local_module;
        expr_id
    }

    fn collect_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> ExprId {
        match expr {
            Some(block) => self.collect_block(block),
            None => self.missing_expr(),
        }
    }

    fn collect_label(&mut self, ast_label: ast::Label) -> LabelId {
        let label = Label {
            name: ast_label.lifetime().as_ref().map_or_else(Name::missing, Name::new_lifetime),
        };
        self.alloc_label(label, AstPtr::new(&ast_label))
    }

    fn collect_pat(&mut self, pat: ast::Pat) -> PatId {
        let pat_id = self.collect_pat_(pat);
        for (_, pats) in self.name_to_pat_grouping.drain() {
            let pats = Arc::<[_]>::from(pats);
            self.body.or_pats.extend(pats.iter().map(|&pat| (pat, pats.clone())));
        }
        self.is_lowering_inside_or_pat = false;
        pat_id
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat),
            None => self.missing_pat(),
        }
    }

    fn collect_pat_(&mut self, pat: ast::Pat) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);

                let key = self.is_lowering_inside_or_pat.then(|| name.clone());
                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat_(subpat));
                let pattern = if annotation == BindingAnnotation::Unannotated && subpat.is_none() {
                    // This could also be a single-segment path pattern. To
                    // decide that, we need to try resolving the name.
                    let (resolved, _) = self.expander.def_map.resolve_path(
                        self.db,
                        self.expander.module,
                        &name.clone().into(),
                        BuiltinShadowMode::Other,
                    );
                    match resolved.take_values() {
                        Some(ModuleDefId::ConstId(_)) => Pat::Path(name.into()),
                        Some(ModuleDefId::EnumVariantId(_)) => {
                            // this is only really valid for unit variants, but
                            // shadowing other enum variants with a pattern is
                            // an error anyway
                            Pat::Path(name.into())
                        }
                        Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                            if self.db.struct_data(s).variant_data.kind() != StructKind::Record =>
                        {
                            // Funnily enough, record structs *can* be shadowed
                            // by pattern bindings (but unit or tuple structs
                            // can't).
                            Pat::Path(name.into())
                        }
                        // shadowing statics is an error as well, so we just ignore that case here
                        _ => Pat::Bind { name, mode: annotation, subpat },
                    }
                } else {
                    Pat::Bind { name, mode: annotation, subpat }
                };

                let ptr = AstPtr::new(&pat);
                let pat = self.alloc_pat(pattern, Either::Left(ptr));
                if let Some(key) = key {
                    self.name_to_pat_grouping.entry(key).or_default().push(pat);
                }
                return pat;
            }
            ast::Pat::TupleStructPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat());
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => {
                self.is_lowering_inside_or_pat = true;
                let pats = p.pats().map(|p| self.collect_pat_(p)).collect();
                Pat::Or(pats)
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt_(p.pat()),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let args = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .fields()
                    .filter_map(|f| {
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat_(ast_pat);
                        let name = f.field_name()?.as_name();
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .rest_pat()
                    .is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                // FIXME properly handle `RestPat`
                Pat::Slice {
                    prefix: prefix.into_iter().map(|p| self.collect_pat_(p)).collect(),
                    slice: slice.map(|p| self.collect_pat_(p)),
                    suffix: suffix.into_iter().map(|p| self.collect_pat_(p)).collect(),
                }
            }
            ast::Pat::LiteralPat(lit) => {
                if let Some(ast_lit) = lit.literal() {
                    let expr = Expr::Literal(ast_lit.kind().into());
                    let expr_ptr = AstPtr::new(&ast::Expr::Literal(ast_lit));
                    let expr_id = self.alloc_expr(expr, expr_ptr);
                    Pat::Lit(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::RestPat(_) => {
                // `RestPat` requires special handling and should not be mapped
                // to a Pat. Here we are using `Pat::Missing` as a fallback for
                // when `RestPat` is mapped to `Pat`, which can easily happen
                // when the source code being analyzed has a malformed pattern
                // which includes `..` in a place where it isn't valid.

                Pat::Missing
            }
            ast::Pat::BoxPat(boxpat) => {
                let inner = self.collect_pat_opt_(boxpat.pat());
                Pat::Box { inner }
            }
            ast::Pat::ConstBlockPat(const_block_pat) => {
                if let Some(expr) = const_block_pat.block_expr() {
                    let expr_id = self.collect_block(expr);
                    Pat::ConstBlock(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::MacroPat(mac) => match mac.macro_call() {
                Some(call) => {
                    let macro_ptr = AstPtr::new(&call);
                    let mut pat = None;
                    self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                        pat = Some(this.collect_pat_opt_(expanded_pat));
                    });

                    match pat {
                        Some(pat) => return pat,
                        None => Pat::Missing,
                    }
                }
                None => Pat::Missing,
            },
            // FIXME: implement
            ast::Pat::RangePat(_) => Pat::Missing,
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, Either::Left(ptr))
    }

    fn collect_pat_opt_(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_pat_(pat),
            None => self.missing_pat(),
        }
    }

    fn collect_tuple_pat(&mut self, args: AstChildren<ast::Pat>) -> (Box<[PatId]>, Option<usize>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis = args.clone().position(|p| matches!(p, ast::Pat::RestPat(_)));
        // We want to skip the `..` pattern here, since we account for it above.
        let args = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.collect_pat_(p))
            .collect();

        (args, ellipsis)
    }

    /// Returns `None` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `Some(())` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::HasAttrs) -> Option<()> {
        match self.expander.parse_attrs(self.db, owner).cfg() {
            Some(cfg) => {
                if self.expander.cfg_options().check(&cfg) != Some(false) {
                    return Some(());
                }

                self.source_map.diagnostics.push(BodyDiagnostic::InactiveCode {
                    node: InFile::new(
                        self.expander.current_file_id,
                        SyntaxNodePtr::new(owner.syntax()),
                    ),
                    cfg,
                    opts: self.expander.cfg_options().clone(),
                });

                None
            }
            None => Some(()),
        }
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        match ast_lit_kind {
            // FIXME: these should have actual values filled in, but unsure on perf impact
            LiteralKind::IntNumber(lit) => {
                if let builtin @ Some(_) = lit.suffix().and_then(BuiltinFloat::from_suffix) {
                    Literal::Float(Default::default(), builtin)
                } else if let builtin @ Some(_) = lit.suffix().and_then(BuiltinInt::from_suffix) {
                    Literal::Int(lit.value().unwrap_or(0) as i128, builtin)
                } else {
                    let builtin = lit.suffix().and_then(BuiltinUint::from_suffix);
                    Literal::Uint(lit.value().unwrap_or(0), builtin)
                }
            }
            LiteralKind::FloatNumber(lit) => {
                let ty = lit.suffix().and_then(BuiltinFloat::from_suffix);
                Literal::Float(Default::default(), ty)
            }
            LiteralKind::ByteString(bs) => {
                let text = bs.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::ByteString(text)
            }
            LiteralKind::String(s) => {
                let text = s.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::String(text)
            }
            LiteralKind::Byte => Literal::Uint(Default::default(), Some(BuiltinUint::U8)),
            LiteralKind::Bool(val) => Literal::Bool(val),
            LiteralKind::Char => Literal::Char(Default::default()),
        }
    }
}
