//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::mem;

use either::Either;
use hir_expand::{
    hygiene::Hygiene,
    name::{name, AsName, Name},
    ExpandError, HirFileId,
};
use la_arena::Arena;
use profile::Count;
use syntax::{
    ast::{
        self, ArgListOwner, ArrayExprKind, AstChildren, LiteralKind, LoopBodyOwner, NameOwner,
        SlicePatComponents,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};

use crate::{
    adt::StructKind,
    body::{Body, BodySourceMap, Expander, LabelSource, PatPtr, SyntheticSyntax},
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint},
    db::DefDatabase,
    diagnostics::{InactiveCode, MacroError, UnresolvedProcMacro},
    expr::{
        dummy_expr_id, ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Label,
        LabelId, Literal, LogicOp, MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField,
        Statement,
    },
    item_scope::BuiltinShadowMode,
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, BlockLoc, ModuleDefId,
};

use super::{diagnostics::BodyDiagnostic, ExprSource, PatSource};

pub(crate) struct LowerCtx {
    hygiene: Hygiene,
}

impl LowerCtx {
    pub(crate) fn new(db: &dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx { hygiene: Hygiene::new(db.upcast(), file_id) }
    }
    pub(crate) fn with_hygiene(hygiene: &Hygiene) -> Self {
        LowerCtx { hygiene: hygiene.clone() }
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(ast, &self.hygiene)
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
        },
        expander,
    }
    .collect(params, body)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    expander: Expander,
    body: Body,
    source_map: BodySourceMap,
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

            for param in param_list.params() {
                let pat = match param.pat() {
                    None => continue,
                    Some(pat) => pat,
                };
                let param_pat = self.collect_pat(pat);
                self.body.params.push(param_pat);
            }
        };

        self.body.body_expr = self.collect_expr_opt(body);
        (self.body, self.source_map)
    }

    fn ctx(&self) -> LowerCtx {
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
    fn unit(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Tuple { exprs: Vec::new() })
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
        let syntax_ptr = AstPtr::new(&expr);
        if self.check_cfg(&expr).is_none() {
            return self.missing_expr();
        }

        match expr {
            ast::Expr::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = match e.condition() {
                    None => self.missing_expr(),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.missing_pat();
                            let arms = vec![
                                MatchArm { pat, expr: then_branch, guard: None },
                                MatchArm {
                                    pat: placeholder_pat,
                                    expr: else_branch.unwrap_or_else(|| self.unit()),
                                    guard: None,
                                },
                            ];
                            return self
                                .alloc_expr(Expr::Match { expr: match_expr, arms }, syntax_ptr);
                        }
                    },
                };

                self.alloc_expr(Expr::If { condition, then_branch, else_branch }, syntax_ptr)
            }
            ast::Expr::EffectExpr(e) => match e.effect() {
                ast::Effect::Try(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::TryBlock { body }, syntax_ptr)
                }
                ast::Effect::Unsafe(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::Unsafe { body }, syntax_ptr)
                }
                // FIXME: we need to record these effects somewhere...
                ast::Effect::Label(label) => {
                    let label = self.collect_label(label);
                    match e.block_expr() {
                        Some(block) => {
                            let res = self.collect_block(block);
                            match &mut self.body.exprs[res] {
                                Expr::Block { label: block_label, .. } => {
                                    *block_label = Some(label);
                                }
                                _ => unreachable!(),
                            }
                            res
                        }
                        None => self.missing_expr(),
                    }
                }
                // FIXME: we need to record these effects somewhere...
                ast::Effect::Async(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::Async { body }, syntax_ptr)
                }
                ast::Effect::Const(_) => {
                    let body = self.collect_block_opt(e.block_expr());
                    self.alloc_expr(Expr::Const { body }, syntax_ptr)
                }
            },
            ast::Expr::BlockExpr(e) => self.collect_block(e),
            ast::Expr::LoopExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_block_opt(e.loop_body());
                self.alloc_expr(Expr::Loop { body, label }, syntax_ptr)
            }
            ast::Expr::WhileExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_block_opt(e.loop_body());

                let condition = match e.condition() {
                    None => self.missing_expr(),
                    Some(condition) => match condition.pat() {
                        None => self.collect_expr_opt(condition.expr()),
                        // if let -- desugar to match
                        Some(pat) => {
                            cov_mark::hit!(infer_resolve_while_let);
                            let pat = self.collect_pat(pat);
                            let match_expr = self.collect_expr_opt(condition.expr());
                            let placeholder_pat = self.missing_pat();
                            let break_ =
                                self.alloc_expr_desugared(Expr::Break { expr: None, label: None });
                            let arms = vec![
                                MatchArm { pat, expr: body, guard: None },
                                MatchArm { pat: placeholder_pat, expr: break_, guard: None },
                            ];
                            let match_expr =
                                self.alloc_expr_desugared(Expr::Match { expr: match_expr, arms });
                            return self
                                .alloc_expr(Expr::Loop { body: match_expr, label }, syntax_ptr);
                        }
                    },
                };

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
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                self.alloc_expr(Expr::Call { callee, args }, syntax_ptr)
            }
            ast::Expr::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.receiver());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().map(|e| self.collect_expr(e)).collect()
                } else {
                    Vec::new()
                };
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args =
                    e.generic_arg_list().and_then(|it| GenericArgs::from_ast(&self.ctx(), it));
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
                                    .and_then(|guard| guard.expr())
                                    .map(|e| self.collect_expr(e)),
                            })
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.expander.parse_path(path))
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
                let path = e.path().and_then(|path| self.expander.parse_path(path));
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
                    Expr::RecordLit { path, fields: Vec::new(), spread: None }
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
                let type_ref = TypeRef::from_ast_opt(&self.ctx(), e.ty());
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
                if let Some(op) = e.op_kind() {
                    self.alloc_expr(Expr::UnaryOp { expr, op }, syntax_ptr)
                } else {
                    self.alloc_expr(Expr::Missing, syntax_ptr)
                }
            }
            ast::Expr::ClosureExpr(e) => {
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    for param in pl.params() {
                        let pat = self.collect_pat_opt(param.pat());
                        let type_ref = param.ty().map(|it| TypeRef::from_ast(&self.ctx(), it));
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let ret_type =
                    e.ret_type().and_then(|r| r.ty()).map(|it| TypeRef::from_ast(&self.ctx(), it));
                let body = self.collect_expr_opt(e.body());
                self.alloc_expr(Expr::Lambda { args, arg_types, ret_type, body }, syntax_ptr)
            }
            ast::Expr::BinExpr(e) => {
                let lhs = self.collect_expr_opt(e.lhs());
                let rhs = self.collect_expr_opt(e.rhs());
                let op = e.op_kind().map(BinaryOp::from);
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
                let mut ids = vec![];
                self.collect_macro_call(e, syntax_ptr.clone(), true, |this, expansion| {
                    ids.push(match expansion {
                        Some(it) => this.collect_expr(it),
                        None => this.alloc_expr(Expr::Missing, syntax_ptr.clone()),
                    })
                });
                ids[0]
            }
            ast::Expr::MacroStmts(e) => {
                // FIXME:  these statements should be held by some hir containter
                for stmt in e.statements() {
                    self.collect_stmt(stmt);
                }
                if let Some(expr) = e.expr() {
                    self.collect_expr(expr)
                } else {
                    self.alloc_expr(Expr::Missing, syntax_ptr)
                }
            }
        }
    }

    fn collect_macro_call<F: FnMut(&mut Self, Option<T>), T: ast::AstNode>(
        &mut self,
        e: ast::MacroCall,
        syntax_ptr: AstPtr<ast::Expr>,
        is_error_recoverable: bool,
        mut collector: F,
    ) {
        // File containing the macro call. Expansion errors will be attached here.
        let outer_file = self.expander.current_file_id;

        let macro_call = self.expander.to_source(AstPtr::new(&e));
        let res = self.expander.enter_expand(self.db, e);

        match &res.err {
            Some(ExpandError::UnresolvedProcMacro) => {
                self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedProcMacro(
                    UnresolvedProcMacro {
                        file: outer_file,
                        node: syntax_ptr.into(),
                        precise_location: None,
                        macro_name: None,
                    },
                ));
            }
            Some(err) => {
                self.source_map.diagnostics.push(BodyDiagnostic::MacroError(MacroError {
                    file: outer_file,
                    node: syntax_ptr.into(),
                    message: err.to_string(),
                }));
            }
            None => {}
        }

        match res.value {
            Some((mark, expansion)) => {
                // FIXME: Statements are too complicated to recover from error for now.
                // It is because we don't have any hygiene for local variable expansion right now.
                if !is_error_recoverable && res.err.is_some() {
                    self.expander.exit(self.db, mark);
                    collector(self, None);
                } else {
                    self.source_map.expansions.insert(macro_call, self.expander.current_file_id);

                    let id = collector(self, Some(expansion));
                    self.expander.exit(self.db, mark);
                    id
                }
            }
            None => collector(self, None),
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        if let Some(expr) = expr {
            self.collect_expr(expr)
        } else {
            self.missing_expr()
        }
    }

    fn collect_stmt(&mut self, s: ast::Stmt) -> Option<Vec<Statement>> {
        let stmt = match s {
            ast::Stmt::LetStmt(stmt) => {
                self.check_cfg(&stmt)?;

                let pat = self.collect_pat_opt(stmt.pat());
                let type_ref = stmt.ty().map(|it| TypeRef::from_ast(&self.ctx(), it));
                let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                vec![Statement::Let { pat, type_ref, initializer }]
            }
            ast::Stmt::ExprStmt(stmt) => {
                self.check_cfg(&stmt)?;

                // Note that macro could be expended to multiple statements
                if let Some(ast::Expr::MacroCall(m)) = stmt.expr() {
                    let syntax_ptr = AstPtr::new(&stmt.expr().unwrap());
                    let mut stmts = vec![];

                    self.collect_macro_call(m, syntax_ptr.clone(), false, |this, expansion| {
                        match expansion {
                            Some(expansion) => {
                                let statements: ast::MacroStmts = expansion;

                                statements.statements().for_each(|stmt| {
                                    if let Some(mut r) = this.collect_stmt(stmt) {
                                        stmts.append(&mut r);
                                    }
                                });
                                if let Some(expr) = statements.expr() {
                                    stmts.push(Statement::Expr(this.collect_expr(expr)));
                                }
                            }
                            None => {
                                stmts.push(Statement::Expr(
                                    this.alloc_expr(Expr::Missing, syntax_ptr.clone()),
                                ));
                            }
                        }
                    });
                    stmts
                } else {
                    vec![Statement::Expr(self.collect_expr_opt(stmt.expr()))]
                }
            }
            ast::Stmt::Item(item) => {
                self.check_cfg(&item)?;

                return None;
            }
        };

        Some(stmt)
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        let ast_id = self.expander.ast_id(&block);
        let block_loc =
            BlockLoc { ast_id, module: self.expander.def_map.module_id(self.expander.module) };
        let block_id = self.db.intern_block(block_loc);
        self.body.block_scopes.push(block_id);

        let opt_def_map = self.db.block_def_map(block_id);
        let has_def_map = opt_def_map.is_some();
        let def_map = opt_def_map.unwrap_or_else(|| self.expander.def_map.clone());
        let module = if has_def_map { def_map.root() } else { self.expander.module };
        let prev_def_map = mem::replace(&mut self.expander.def_map, def_map);
        let prev_local_module = mem::replace(&mut self.expander.module, module);

        let statements =
            block.statements().filter_map(|s| self.collect_stmt(s)).flatten().collect();
        let tail = block.tail_expr().map(|e| self.collect_expr(e));
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
        if let Some(block) = expr {
            self.collect_block(block)
        } else {
            self.missing_expr()
        }
    }

    fn collect_label(&mut self, ast_label: ast::Label) -> LabelId {
        let label = Label {
            name: ast_label.lifetime().as_ref().map_or_else(Name::missing, Name::new_lifetime),
        };
        self.alloc_label(label, AstPtr::new(&ast_label))
    }

    fn collect_pat(&mut self, pat: ast::Pat) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat(subpat));
                if annotation == BindingAnnotation::Unannotated && subpat.is_none() {
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
                }
            }
            ast::Pat::TupleStructPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat());
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => {
                let pats = p.pats().map(|p| self.collect_pat(p)).collect();
                Pat::Or(pats)
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt(p.pat()),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(p.fields());
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path = p.path().and_then(|path| self.expander.parse_path(path));
                let args: Vec<_> = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .fields()
                    .filter_map(|f| {
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat(ast_pat);
                        let name = f.field_name()?.as_name();
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .dotdot_token()
                    .is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                // FIXME properly handle `RestPat`
                Pat::Slice {
                    prefix: prefix.into_iter().map(|p| self.collect_pat(p)).collect(),
                    slice: slice.map(|p| self.collect_pat(p)),
                    suffix: suffix.into_iter().map(|p| self.collect_pat(p)).collect(),
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
                let inner = self.collect_pat_opt(boxpat.pat());
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
            // FIXME: implement
            ast::Pat::RangePat(_) | ast::Pat::MacroPat(_) => Pat::Missing,
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, Either::Left(ptr))
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        if let Some(pat) = pat {
            self.collect_pat(pat)
        } else {
            self.missing_pat()
        }
    }

    fn collect_tuple_pat(&mut self, args: AstChildren<ast::Pat>) -> (Vec<PatId>, Option<usize>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis = args.clone().position(|p| matches!(p, ast::Pat::RestPat(_)));
        // We want to skip the `..` pattern here, since we account for it above.
        let args = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.collect_pat(p))
            .collect();

        (args, ellipsis)
    }

    /// Returns `None` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `Some(())` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::AttrsOwner) -> Option<()> {
        match self.expander.parse_attrs(self.db, owner).cfg() {
            Some(cfg) => {
                if self.expander.cfg_options().check(&cfg) != Some(false) {
                    return Some(());
                }

                self.source_map.diagnostics.push(BodyDiagnostic::InactiveCode(InactiveCode {
                    file: self.expander.current_file_id,
                    node: SyntaxNodePtr::new(owner.syntax()),
                    cfg,
                    opts: self.expander.cfg_options().clone(),
                }));

                None
            }
            None => Some(()),
        }
    }
}

impl From<ast::BinOp> for BinaryOp {
    fn from(ast_op: ast::BinOp) -> Self {
        match ast_op {
            ast::BinOp::BooleanOr => BinaryOp::LogicOp(LogicOp::Or),
            ast::BinOp::BooleanAnd => BinaryOp::LogicOp(LogicOp::And),
            ast::BinOp::EqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: false }),
            ast::BinOp::NegatedEqualityTest => BinaryOp::CmpOp(CmpOp::Eq { negated: true }),
            ast::BinOp::LesserEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: false })
            }
            ast::BinOp::GreaterEqualTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: false })
            }
            ast::BinOp::LesserTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Less, strict: true })
            }
            ast::BinOp::GreaterTest => {
                BinaryOp::CmpOp(CmpOp::Ord { ordering: Ordering::Greater, strict: true })
            }
            ast::BinOp::Addition => BinaryOp::ArithOp(ArithOp::Add),
            ast::BinOp::Multiplication => BinaryOp::ArithOp(ArithOp::Mul),
            ast::BinOp::Subtraction => BinaryOp::ArithOp(ArithOp::Sub),
            ast::BinOp::Division => BinaryOp::ArithOp(ArithOp::Div),
            ast::BinOp::Remainder => BinaryOp::ArithOp(ArithOp::Rem),
            ast::BinOp::LeftShift => BinaryOp::ArithOp(ArithOp::Shl),
            ast::BinOp::RightShift => BinaryOp::ArithOp(ArithOp::Shr),
            ast::BinOp::BitwiseXor => BinaryOp::ArithOp(ArithOp::BitXor),
            ast::BinOp::BitwiseOr => BinaryOp::ArithOp(ArithOp::BitOr),
            ast::BinOp::BitwiseAnd => BinaryOp::ArithOp(ArithOp::BitAnd),
            ast::BinOp::Assignment => BinaryOp::Assignment { op: None },
            ast::BinOp::AddAssign => BinaryOp::Assignment { op: Some(ArithOp::Add) },
            ast::BinOp::DivAssign => BinaryOp::Assignment { op: Some(ArithOp::Div) },
            ast::BinOp::MulAssign => BinaryOp::Assignment { op: Some(ArithOp::Mul) },
            ast::BinOp::RemAssign => BinaryOp::Assignment { op: Some(ArithOp::Rem) },
            ast::BinOp::ShlAssign => BinaryOp::Assignment { op: Some(ArithOp::Shl) },
            ast::BinOp::ShrAssign => BinaryOp::Assignment { op: Some(ArithOp::Shr) },
            ast::BinOp::SubAssign => BinaryOp::Assignment { op: Some(ArithOp::Sub) },
            ast::BinOp::BitOrAssign => BinaryOp::Assignment { op: Some(ArithOp::BitOr) },
            ast::BinOp::BitAndAssign => BinaryOp::Assignment { op: Some(ArithOp::BitAnd) },
            ast::BinOp::BitXorAssign => BinaryOp::Assignment { op: Some(ArithOp::BitXor) },
        }
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        match ast_lit_kind {
            LiteralKind::IntNumber(lit) => {
                if let builtin @ Some(_) = lit.suffix().and_then(BuiltinFloat::from_suffix) {
                    return Literal::Float(Default::default(), builtin);
                } else if let builtin @ Some(_) =
                    lit.suffix().and_then(|it| BuiltinInt::from_suffix(&it))
                {
                    Literal::Int(Default::default(), builtin)
                } else {
                    let builtin = lit.suffix().and_then(|it| BuiltinUint::from_suffix(&it));
                    Literal::Uint(Default::default(), builtin)
                }
            }
            LiteralKind::FloatNumber(lit) => {
                let ty = lit.suffix().and_then(|it| BuiltinFloat::from_suffix(&it));
                Literal::Float(Default::default(), ty)
            }
            LiteralKind::ByteString(_) => Literal::ByteString(Default::default()),
            LiteralKind::String(_) => Literal::String(Default::default()),
            LiteralKind::Byte => Literal::Uint(Default::default(), Some(BuiltinUint::U8)),
            LiteralKind::Bool(val) => Literal::Bool(val),
            LiteralKind::Char => Literal::Char(Default::default()),
        }
    }
}
