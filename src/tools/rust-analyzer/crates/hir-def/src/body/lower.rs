//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::{mem, sync::Arc};

use either::Either;
use hir_expand::{
    ast_id_map::AstIdMap,
    hygiene::Hygiene,
    name::{name, AsName, Name},
    AstId, ExpandError, HirFileId, InFile,
};
use intern::Interned;
use la_arena::Arena;
use once_cell::unsync::OnceCell;
use profile::Count;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use syntax::{
    ast::{
        self, ArrayExprKind, AstChildren, HasArgList, HasLoopBody, HasName, LiteralKind,
        SlicePatComponents,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};

use crate::{
    adt::StructKind,
    body::{Body, BodySourceMap, Expander, ExprPtr, LabelPtr, LabelSource, PatPtr},
    body::{BodyDiagnostic, ExprSource, PatSource},
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint},
    db::DefDatabase,
    expr::{
        dummy_expr_id, Array, Binding, BindingAnnotation, BindingId, ClosureKind, Expr, ExprId,
        FloatTypeWrapper, Label, LabelId, Literal, MatchArm, Movability, Pat, PatId,
        RecordFieldPat, RecordLitField, Statement,
    },
    item_scope::BuiltinShadowMode,
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, BlockId, BlockLoc, ModuleDefId, UnresolvedMacro,
};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    hygiene: Hygiene,
    ast_id_map: Option<(HirFileId, OnceCell<Arc<AstIdMap>>)>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx {
            db,
            hygiene: Hygiene::new(db.upcast(), file_id),
            ast_id_map: Some((file_id, OnceCell::new())),
        }
    }

    pub fn with_hygiene(db: &'a dyn DefDatabase, hygiene: &Hygiene) -> Self {
        LowerCtx { db, hygiene: hygiene.clone(), ast_id_map: None }
    }

    pub(crate) fn hygiene(&self) -> &Hygiene {
        &self.hygiene
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(ast, self)
    }

    pub(crate) fn ast_id<N: AstNode>(&self, item: &N) -> Option<AstId<N>> {
        let &(file_id, ref ast_id_map) = self.ast_id_map.as_ref()?;
        let ast_id_map = ast_id_map.get_or_init(|| self.db.ast_id_map(file_id));
        Some(InFile::new(file_id, ast_id_map.ast_id(item)))
    }
}

pub(super) fn lower(
    db: &dyn DefDatabase,
    expander: Expander,
    params: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
    body: Option<ast::Expr>,
) -> (Body, BodySourceMap) {
    ExprCollector {
        db,
        source_map: BodySourceMap::default(),
        ast_id_map: db.ast_id_map(expander.current_file_id),
        body: Body {
            exprs: Arena::default(),
            pats: Arena::default(),
            bindings: Arena::default(),
            labels: Arena::default(),
            params: Vec::new(),
            body_expr: dummy_expr_id(),
            block_scopes: Vec::new(),
            _c: Count::new(),
        },
        expander,
        is_lowering_assignee_expr: false,
        is_lowering_generator: false,
    }
    .collect(params, body)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    expander: Expander,
    ast_id_map: Arc<AstIdMap>,
    body: Body,
    source_map: BodySourceMap,
    is_lowering_assignee_expr: bool,
    is_lowering_generator: bool,
}

#[derive(Debug, Default)]
struct BindingList {
    map: FxHashMap<Name, BindingId>,
}

impl BindingList {
    fn find(
        &mut self,
        ec: &mut ExprCollector<'_>,
        name: Name,
        mode: BindingAnnotation,
    ) -> BindingId {
        *self.map.entry(name).or_insert_with_key(|n| ec.alloc_binding(n.clone(), mode))
    }
}

impl ExprCollector<'_> {
    fn collect(
        mut self,
        param_list: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
        body: Option<ast::Expr>,
    ) -> (Body, BodySourceMap) {
        if let Some((param_list, mut attr_enabled)) = param_list {
            if let Some(self_param) =
                param_list.self_param().filter(|_| attr_enabled.next().unwrap_or(false))
            {
                let ptr = AstPtr::new(&self_param);
                let binding_id = self.alloc_binding(
                    name![self],
                    BindingAnnotation::new(
                        self_param.mut_token().is_some() && self_param.amp_token().is_none(),
                        false,
                    ),
                );
                let param_pat =
                    self.alloc_pat(Pat::Bind { id: binding_id, subpat: None }, Either::Right(ptr));
                self.add_definition_to_binding(binding_id, param_pat);
                self.body.params.push(param_pat);
            }

            for pat in param_list
                .params()
                .zip(attr_enabled)
                .filter_map(|(param, enabled)| param.pat().filter(|_| enabled))
            {
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

    fn alloc_expr(&mut self, expr: Expr, ptr: ExprPtr) -> ExprId {
        let src = self.expander.to_source(ptr);
        let id = self.make_expr(expr, src.clone());
        self.source_map.expr_map.insert(src, id);
        id
    }
    // desugared exprs don't have ptr, that's wrong and should be fixed
    // somehow.
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.body.exprs.alloc(expr)
    }
    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }
    fn make_expr(&mut self, expr: Expr, src: ExprSource) -> ExprId {
        let id = self.body.exprs.alloc(expr);
        self.source_map.expr_map_back.insert(id, src);
        id
    }

    fn alloc_binding(&mut self, name: Name, mode: BindingAnnotation) -> BindingId {
        self.body.bindings.alloc(Binding { name, mode, definitions: SmallVec::new() })
    }
    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.to_source(ptr);
        let id = self.make_pat(pat, src.clone());
        self.source_map.pat_map.insert(src, id);
        id
    }
    fn missing_pat(&mut self) -> PatId {
        self.body.pats.alloc(Pat::Missing)
    }
    fn make_pat(&mut self, pat: Pat, src: PatSource) -> PatId {
        let id = self.body.pats.alloc(pat);
        self.source_map.pat_map_back.insert(id, src);
        id
    }

    fn alloc_label(&mut self, label: Label, ptr: LabelPtr) -> LabelId {
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
                    self.collect_block_(e, |id, statements, tail| Expr::TryBlock {
                        id,
                        statements,
                        tail,
                    })
                }
                Some(ast::BlockModifier::Unsafe(_)) => {
                    self.collect_block_(e, |id, statements, tail| Expr::Unsafe {
                        id,
                        statements,
                        tail,
                    })
                }
                Some(ast::BlockModifier::Label(label)) => {
                    let label = self.collect_label(label);
                    self.collect_block_(e, |id, statements, tail| Expr::Block {
                        id,
                        statements,
                        tail,
                        label: Some(label),
                    })
                }
                Some(ast::BlockModifier::Async(_)) => self
                    .collect_block_(e, |id, statements, tail| Expr::Async { id, statements, tail }),
                Some(ast::BlockModifier::Const(_)) => self
                    .collect_block_(e, |id, statements, tail| Expr::Const { id, statements, tail }),
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
                self.alloc_expr(
                    Expr::Call { callee, args, is_assignee_expr: self.is_lowering_assignee_expr },
                    syntax_ptr,
                )
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
                self.is_lowering_generator = true;
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Yield { expr }, syntax_ptr)
            }
            ast::Expr::YeetExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Yeet { expr }, syntax_ptr)
            }
            ast::Expr::RecordExpr(e) => {
                let path =
                    e.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let is_assignee_expr = self.is_lowering_assignee_expr;
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
                    let ellipsis = nfl.dotdot_token().is_some();
                    Expr::RecordLit { path, fields, spread, ellipsis, is_assignee_expr }
                } else {
                    Expr::RecordLit {
                        path,
                        fields: Box::default(),
                        spread: None,
                        ellipsis: false,
                        is_assignee_expr,
                    }
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

                let prev_is_lowering_generator = self.is_lowering_generator;
                self.is_lowering_generator = false;

                let body = self.collect_expr_opt(e.body());

                let closure_kind = if self.is_lowering_generator {
                    let movability = if e.static_token().is_some() {
                        Movability::Static
                    } else {
                        Movability::Movable
                    };
                    ClosureKind::Generator(movability)
                } else {
                    ClosureKind::Closure
                };
                self.is_lowering_generator = prev_is_lowering_generator;

                self.alloc_expr(
                    Expr::Closure {
                        args: args.into(),
                        arg_types: arg_types.into(),
                        ret_type,
                        body,
                        closure_kind,
                    },
                    syntax_ptr,
                )
            }
            ast::Expr::BinExpr(e) => {
                let op = e.op_kind();
                if let Some(ast::BinaryOp::Assignment { op: None }) = op {
                    self.is_lowering_assignee_expr = true;
                }
                let lhs = self.collect_expr_opt(e.lhs());
                self.is_lowering_assignee_expr = false;
                let rhs = self.collect_expr_opt(e.rhs());
                self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
            }
            ast::Expr::TupleExpr(e) => {
                let exprs = e.fields().map(|expr| self.collect_expr(expr)).collect();
                self.alloc_expr(
                    Expr::Tuple { exprs, is_assignee_expr: self.is_lowering_assignee_expr },
                    syntax_ptr,
                )
            }
            ast::Expr::BoxExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Box { expr }, syntax_ptr)
            }

            ast::Expr::ArrayExpr(e) => {
                let kind = e.kind();

                match kind {
                    ArrayExprKind::ElementList(e) => {
                        let elements = e.map(|expr| self.collect_expr(expr)).collect();
                        self.alloc_expr(
                            Expr::Array(Array::ElementList {
                                elements,
                                is_assignee_expr: self.is_lowering_assignee_expr,
                            }),
                            syntax_ptr,
                        )
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
            ast::Expr::MacroExpr(e) => {
                let e = e.macro_call()?;
                let macro_ptr = AstPtr::new(&e);
                let id = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    expansion.map(|it| this.collect_expr(it))
                });
                match id {
                    Some(id) => {
                        // Make the macro-call point to its expanded expression so we can query
                        // semantics on syntax pointers to the macro
                        let src = self.expander.to_source(syntax_ptr);
                        self.source_map.expr_map.insert(src, id);
                        id
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::UnderscoreExpr(_) => self.alloc_expr(Expr::Underscore, syntax_ptr),
        })
    }

    fn collect_macro_call<F, T, U>(
        &mut self,
        mcall: ast::MacroCall,
        syntax_ptr: AstPtr<ast::MacroCall>,
        record_diagnostics: bool,
        collector: F,
    ) -> U
    where
        F: FnOnce(&mut Self, Option<T>) -> U,
        T: ast::AstNode,
    {
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
                return collector(self, None);
            }
        };

        if record_diagnostics {
            match &res.err {
                Some(ExpandError::UnresolvedProcMacro(krate)) => {
                    self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedProcMacro {
                        node: InFile::new(outer_file, syntax_ptr),
                        krate: *krate,
                    });
                }
                Some(ExpandError::RecursionOverflowPosioned) => {
                    // Recursion limit has been reached in the macro expansion tree, but not in
                    // this very macro call. Don't add diagnostics to avoid duplication.
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
                // Keep collecting even with expansion errors so we can provide completions and
                // other services in incomplete macro expressions.
                self.source_map.expansions.insert(macro_call_ptr, self.expander.current_file_id);
                let prev_ast_id_map = mem::replace(
                    &mut self.ast_id_map,
                    self.db.ast_id_map(self.expander.current_file_id),
                );

                let id = collector(self, Some(expansion));
                self.ast_id_map = prev_ast_id_map;
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

    fn collect_macro_as_stmt(
        &mut self,
        statements: &mut Vec<Statement>,
        mac: ast::MacroExpr,
    ) -> Option<ExprId> {
        let mac_call = mac.macro_call()?;
        let syntax_ptr = AstPtr::new(&ast::Expr::from(mac));
        let macro_ptr = AstPtr::new(&mac_call);
        let expansion = self.collect_macro_call(
            mac_call,
            macro_ptr,
            false,
            |this, expansion: Option<ast::MacroStmts>| match expansion {
                Some(expansion) => {
                    expansion.statements().for_each(|stmt| this.collect_stmt(statements, stmt));
                    expansion.expr().and_then(|expr| match expr {
                        ast::Expr::MacroExpr(mac) => this.collect_macro_as_stmt(statements, mac),
                        expr => Some(this.collect_expr(expr)),
                    })
                }
                None => None,
            },
        );
        match expansion {
            Some(tail) => {
                // Make the macro-call point to its expanded expression so we can query
                // semantics on syntax pointers to the macro
                let src = self.expander.to_source(syntax_ptr);
                self.source_map.expr_map.insert(src, tail);
                Some(tail)
            }
            None => None,
        }
    }

    fn collect_stmt(&mut self, statements: &mut Vec<Statement>, s: ast::Stmt) {
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
                statements.push(Statement::Let { pat, type_ref, initializer, else_branch });
            }
            ast::Stmt::ExprStmt(stmt) => {
                let expr = stmt.expr();
                match &expr {
                    Some(expr) if self.check_cfg(expr).is_none() => return,
                    _ => (),
                }
                let has_semi = stmt.semicolon_token().is_some();
                // Note that macro could be expanded to multiple statements
                if let Some(ast::Expr::MacroExpr(mac)) = expr {
                    if let Some(expr) = self.collect_macro_as_stmt(statements, mac) {
                        statements.push(Statement::Expr { expr, has_semi })
                    }
                } else {
                    let expr = self.collect_expr_opt(expr);
                    statements.push(Statement::Expr { expr, has_semi });
                }
            }
            ast::Stmt::Item(_item) => (),
        }
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        self.collect_block_(block, |id, statements, tail| Expr::Block {
            id,
            statements,
            tail,
            label: None,
        })
    }

    fn collect_block_(
        &mut self,
        block: ast::BlockExpr,
        mk_block: impl FnOnce(BlockId, Box<[Statement]>, Option<ExprId>) -> Expr,
    ) -> ExprId {
        let file_local_id = self.ast_id_map.ast_id(&block);
        let ast_id = AstId::new(self.expander.current_file_id, file_local_id);
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

        let mut statements = Vec::new();
        block.statements().for_each(|s| self.collect_stmt(&mut statements, s));
        let tail = block.tail_expr().and_then(|e| match e {
            ast::Expr::MacroExpr(mac) => self.collect_macro_as_stmt(&mut statements, mac),
            expr => self.maybe_collect_expr(expr),
        });
        let tail = tail.or_else(|| {
            let stmt = statements.pop()?;
            if let Statement::Expr { expr, has_semi: false } = stmt {
                return Some(expr);
            }
            statements.push(stmt);
            None
        });

        let syntax_node_ptr = AstPtr::new(&block.into());
        let expr_id = self
            .alloc_expr(mk_block(block_id, statements.into_boxed_slice(), tail), syntax_node_ptr);

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
        self.collect_pat_(pat, &mut BindingList::default())
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat),
            None => self.missing_pat(),
        }
    }

    fn collect_pat_(&mut self, pat: ast::Pat, binding_list: &mut BindingList) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);

                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat_(subpat, binding_list));

                let is_simple_ident_pat =
                    annotation == BindingAnnotation::Unannotated && subpat.is_none();
                let (binding, pattern) = if is_simple_ident_pat {
                    // This could also be a single-segment path pattern. To
                    // decide that, we need to try resolving the name.
                    let (resolved, _) = self.expander.def_map.resolve_path(
                        self.db,
                        self.expander.module,
                        &name.clone().into(),
                        BuiltinShadowMode::Other,
                    );
                    match resolved.take_values() {
                        Some(ModuleDefId::ConstId(_)) => (None, Pat::Path(name.into())),
                        Some(ModuleDefId::EnumVariantId(_)) => {
                            // this is only really valid for unit variants, but
                            // shadowing other enum variants with a pattern is
                            // an error anyway
                            (None, Pat::Path(name.into()))
                        }
                        Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                            if self.db.struct_data(s).variant_data.kind() != StructKind::Record =>
                        {
                            // Funnily enough, record structs *can* be shadowed
                            // by pattern bindings (but unit or tuple structs
                            // can't).
                            (None, Pat::Path(name.into()))
                        }
                        // shadowing statics is an error as well, so we just ignore that case here
                        _ => {
                            let id = binding_list.find(self, name, annotation);
                            (Some(id), Pat::Bind { id, subpat })
                        }
                    }
                } else {
                    let id = binding_list.find(self, name, annotation);
                    (Some(id), Pat::Bind { id, subpat })
                };

                let ptr = AstPtr::new(&pat);
                let pat = self.alloc_pat(pattern, Either::Left(ptr));
                if let Some(binding_id) = binding {
                    self.add_definition_to_binding(binding_id, pat);
                }
                return pat;
            }
            ast::Pat::TupleStructPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let (args, ellipsis) = self.collect_tuple_pat(p.fields(), binding_list);
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt_(p.pat(), binding_list);
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => {
                let pats = p.pats().map(|p| self.collect_pat_(p, binding_list)).collect();
                Pat::Or(pats)
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt_(p.pat(), binding_list),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(p.fields(), binding_list);
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
                        let pat = self.collect_pat_(ast_pat, binding_list);
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
                    prefix: prefix
                        .into_iter()
                        .map(|p| self.collect_pat_(p, binding_list))
                        .collect(),
                    slice: slice.map(|p| self.collect_pat_(p, binding_list)),
                    suffix: suffix
                        .into_iter()
                        .map(|p| self.collect_pat_(p, binding_list))
                        .collect(),
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
                let inner = self.collect_pat_opt_(boxpat.pat(), binding_list);
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
                    let src = self.expander.to_source(Either::Left(AstPtr::new(&pat)));
                    let pat =
                        self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                            this.collect_pat_opt_(expanded_pat, binding_list)
                        });
                    self.source_map.pat_map.insert(src, pat);
                    return pat;
                }
                None => Pat::Missing,
            },
            // FIXME: implement
            ast::Pat::RangePat(_) => Pat::Missing,
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, Either::Left(ptr))
    }

    fn collect_pat_opt_(&mut self, pat: Option<ast::Pat>, binding_list: &mut BindingList) -> PatId {
        match pat {
            Some(pat) => self.collect_pat_(pat, binding_list),
            None => self.missing_pat(),
        }
    }

    fn collect_tuple_pat(
        &mut self,
        args: AstChildren<ast::Pat>,
        binding_list: &mut BindingList,
    ) -> (Box<[PatId]>, Option<usize>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis = args.clone().position(|p| matches!(p, ast::Pat::RestPat(_)));
        // We want to skip the `..` pattern here, since we account for it above.
        let args = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.collect_pat_(p, binding_list))
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

    fn add_definition_to_binding(&mut self, binding_id: BindingId, pat_id: PatId) {
        self.body.bindings[binding_id].definitions.push(pat_id);
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        match ast_lit_kind {
            // FIXME: these should have actual values filled in, but unsure on perf impact
            LiteralKind::IntNumber(lit) => {
                if let builtin @ Some(_) = lit.suffix().and_then(BuiltinFloat::from_suffix) {
                    Literal::Float(
                        FloatTypeWrapper::new(lit.float_value().unwrap_or(Default::default())),
                        builtin,
                    )
                } else if let builtin @ Some(_) = lit.suffix().and_then(BuiltinInt::from_suffix) {
                    Literal::Int(lit.value().unwrap_or(0) as i128, builtin)
                } else {
                    let builtin = lit.suffix().and_then(BuiltinUint::from_suffix);
                    Literal::Uint(lit.value().unwrap_or(0), builtin)
                }
            }
            LiteralKind::FloatNumber(lit) => {
                let ty = lit.suffix().and_then(BuiltinFloat::from_suffix);
                Literal::Float(FloatTypeWrapper::new(lit.value().unwrap_or(Default::default())), ty)
            }
            LiteralKind::ByteString(bs) => {
                let text = bs.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::ByteString(text)
            }
            LiteralKind::String(s) => {
                let text = s.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::String(text)
            }
            LiteralKind::Byte(b) => {
                Literal::Uint(b.value().unwrap_or_default() as u128, Some(BuiltinUint::U8))
            }
            LiteralKind::Char(c) => Literal::Char(c.value().unwrap_or_default()),
            LiteralKind::Bool(val) => Literal::Bool(val),
        }
    }
}
