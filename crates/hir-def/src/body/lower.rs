//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::mem;

use base_db::CrateId;
use either::Either;
use hir_expand::{
    ast_id_map::AstIdMap,
    name::{name, AsName, Name},
    ExpandError, InFile,
};
use intern::Interned;
use profile::Count;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use syntax::{
    ast::{
        self, ArrayExprKind, AstChildren, BlockExpr, HasArgList, HasAttrs, HasLoopBody, HasName,
        RangeItem, SlicePatComponents,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};
use triomphe::Arc;

use crate::{
    body::{Body, BodyDiagnostic, BodySourceMap, ExprPtr, LabelPtr, PatPtr},
    builtin_type::BuiltinUint,
    data::adt::StructKind,
    db::DefDatabase,
    expander::Expander,
    hir::{
        dummy_expr_id,
        format_args::{
            self, FormatAlignment, FormatArgs, FormatArgsPiece, FormatArgument, FormatArgumentKind,
            FormatArgumentsCollector, FormatCount, FormatDebugHex, FormatOptions,
            FormatPlaceholder, FormatSign, FormatTrait,
        },
        Array, Binding, BindingAnnotation, BindingId, BindingProblems, CaptureBy, ClosureKind,
        Expr, ExprId, InlineAsm, Label, LabelId, Literal, LiteralOrConst, MatchArm, Movability,
        OffsetOf, Pat, PatId, RecordFieldPat, RecordLitField, Statement,
    },
    item_scope::BuiltinShadowMode,
    lang_item::LangItem,
    lower::LowerCtx,
    nameres::{DefMap, MacroSubNs},
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, BlockId, BlockLoc, ConstBlockLoc, DefWithBodyId, ModuleDefId, UnresolvedMacro,
};

type FxIndexSet<K> = indexmap::IndexSet<K, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

pub(super) fn lower(
    db: &dyn DefDatabase,
    owner: DefWithBodyId,
    expander: Expander,
    params: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
    body: Option<ast::Expr>,
    krate: CrateId,
    is_async_fn: bool,
) -> (Body, BodySourceMap) {
    ExprCollector {
        db,
        owner,
        krate,
        def_map: expander.module.def_map(db),
        source_map: BodySourceMap::default(),
        ast_id_map: db.ast_id_map(expander.current_file_id()),
        body: Body {
            exprs: Default::default(),
            pats: Default::default(),
            bindings: Default::default(),
            binding_owners: Default::default(),
            labels: Default::default(),
            params: Vec::new(),
            body_expr: dummy_expr_id(),
            block_scopes: Vec::new(),
            _c: Count::new(),
        },
        expander,
        current_try_block_label: None,
        is_lowering_assignee_expr: false,
        is_lowering_coroutine: false,
        label_ribs: Vec::new(),
        current_binding_owner: None,
    }
    .collect(params, body, is_async_fn)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    expander: Expander,
    owner: DefWithBodyId,
    def_map: Arc<DefMap>,
    ast_id_map: Arc<AstIdMap>,
    krate: CrateId,
    body: Body,
    source_map: BodySourceMap,

    is_lowering_assignee_expr: bool,
    is_lowering_coroutine: bool,

    current_try_block_label: Option<LabelId>,
    // points to the expression that a try expression will target (replaces current_try_block_label)
    // catch_scope: Option<ExprId>,
    // points to the expression that an unlabeled control flow will target
    // loop_scope: Option<ExprId>,
    // needed to diagnose non label control flow in while conditions
    // is_in_loop_condition: bool,

    // resolution
    label_ribs: Vec<LabelRib>,
    current_binding_owner: Option<ExprId>,
}

#[derive(Clone, Debug)]
struct LabelRib {
    kind: RibKind,
    // Once we handle macro hygiene this will need to be a map
    label: Option<(Name, LabelId)>,
}

impl LabelRib {
    fn new(kind: RibKind) -> Self {
        LabelRib { kind, label: None }
    }
    fn new_normal(label: (Name, LabelId)) -> Self {
        LabelRib { kind: RibKind::Normal, label: Some(label) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RibKind {
    Normal,
    Closure,
    Constant,
}

impl RibKind {
    /// This rib forbids referring to labels defined in upwards ribs.
    fn is_label_barrier(self) -> bool {
        match self {
            RibKind::Normal => false,
            RibKind::Closure | RibKind::Constant => true,
        }
    }
}

#[derive(Debug, Default)]
struct BindingList {
    map: FxHashMap<Name, BindingId>,
    is_used: FxHashMap<BindingId, bool>,
    reject_new: bool,
}

impl BindingList {
    fn find(
        &mut self,
        ec: &mut ExprCollector<'_>,
        name: Name,
        mode: BindingAnnotation,
    ) -> BindingId {
        let id = *self.map.entry(name).or_insert_with_key(|n| ec.alloc_binding(n.clone(), mode));
        if ec.body.bindings[id].mode != mode {
            ec.body.bindings[id].problems = Some(BindingProblems::BoundInconsistently);
        }
        self.check_is_used(ec, id);
        id
    }

    fn check_is_used(&mut self, ec: &mut ExprCollector<'_>, id: BindingId) {
        match self.is_used.get(&id) {
            None => {
                if self.reject_new {
                    ec.body.bindings[id].problems = Some(BindingProblems::NotBoundAcrossAll);
                }
            }
            Some(true) => {
                ec.body.bindings[id].problems = Some(BindingProblems::BoundMoreThanOnce);
            }
            Some(false) => {}
        }
        self.is_used.insert(id, true);
    }
}

impl ExprCollector<'_> {
    fn collect(
        mut self,
        param_list: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
        body: Option<ast::Expr>,
        is_async_fn: bool,
    ) -> (Body, BodySourceMap) {
        if let Some((param_list, mut attr_enabled)) = param_list {
            if let Some(self_param) =
                param_list.self_param().filter(|_| attr_enabled.next().unwrap_or(false))
            {
                let is_mutable =
                    self_param.mut_token().is_some() && self_param.amp_token().is_none();
                let ptr = AstPtr::new(&Either::Right(self_param));
                let binding_id: la_arena::Idx<Binding> =
                    self.alloc_binding(name![self], BindingAnnotation::new(is_mutable, false));
                let param_pat = self.alloc_pat(Pat::Bind { id: binding_id, subpat: None }, ptr);
                self.add_definition_to_binding(binding_id, param_pat);
                self.body.params.push(param_pat);
            }

            for (param, _) in param_list.params().zip(attr_enabled).filter(|(_, enabled)| *enabled)
            {
                let param_pat = self.collect_pat_top(param.pat());
                self.body.params.push(param_pat);
            }
        };
        self.body.body_expr = self.with_label_rib(RibKind::Closure, |this| {
            if is_async_fn {
                match body {
                    Some(e) => {
                        let expr = this.collect_expr(e);
                        this.alloc_expr_desugared(Expr::Async {
                            id: None,
                            statements: Box::new([]),
                            tail: Some(expr),
                        })
                    }
                    None => this.missing_expr(),
                }
            } else {
                this.collect_expr_opt(body)
            }
        });

        (self.body, self.source_map)
    }

    fn ctx(&self) -> LowerCtx<'_> {
        self.expander.ctx(self.db)
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        self.maybe_collect_expr(expr).unwrap_or_else(|| self.missing_expr())
    }

    /// Returns `None` if and only if the expression is `#[cfg]`d out.
    fn maybe_collect_expr(&mut self, expr: ast::Expr) -> Option<ExprId> {
        let syntax_ptr = AstPtr::new(&expr);
        self.check_cfg(&expr)?;

        // FIXME: Move some of these arms out into separate methods for clarity
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
                let pat = self.collect_pat_top(e.pat());
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Let { pat, expr }, syntax_ptr)
            }
            ast::Expr::BlockExpr(e) => match e.modifier() {
                Some(ast::BlockModifier::Try(_)) => self.desugar_try_block(e),
                Some(ast::BlockModifier::Unsafe(_)) => {
                    self.collect_block_(e, |id, statements, tail| Expr::Unsafe {
                        id,
                        statements,
                        tail,
                    })
                }
                Some(ast::BlockModifier::Label(label)) => {
                    let label = self.collect_label(label);
                    self.with_labeled_rib(label, |this| {
                        this.collect_block_(e, |id, statements, tail| Expr::Block {
                            id,
                            statements,
                            tail,
                            label: Some(label),
                        })
                    })
                }
                Some(ast::BlockModifier::Async(_)) => {
                    self.with_label_rib(RibKind::Closure, |this| {
                        this.collect_block_(e, |id, statements, tail| Expr::Async {
                            id,
                            statements,
                            tail,
                        })
                    })
                }
                Some(ast::BlockModifier::Const(_)) => {
                    self.with_label_rib(RibKind::Constant, |this| {
                        let (result_expr_id, prev_binding_owner) =
                            this.initialize_binding_owner(syntax_ptr);
                        let inner_expr = this.collect_block(e);
                        let it = this.db.intern_anonymous_const(ConstBlockLoc {
                            parent: this.owner,
                            root: inner_expr,
                        });
                        this.body.exprs[result_expr_id] = Expr::Const(it);
                        this.current_binding_owner = prev_binding_owner;
                        result_expr_id
                    })
                }
                None => self.collect_block(e),
            },
            ast::Expr::LoopExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_labelled_block_opt(label, e.loop_body());
                self.alloc_expr(Expr::Loop { body, label }, syntax_ptr)
            }
            ast::Expr::WhileExpr(e) => self.collect_while_loop(syntax_ptr, e),
            ast::Expr::ForExpr(e) => self.collect_for_loop(syntax_ptr, e),
            ast::Expr::CallExpr(e) => {
                let is_rustc_box = {
                    let attrs = e.attrs();
                    attrs.filter_map(|it| it.as_simple_atom()).any(|it| it == "rustc_box")
                };
                if is_rustc_box {
                    let expr = self.collect_expr_opt(e.arg_list().and_then(|it| it.args().next()));
                    self.alloc_expr(Expr::Box { expr }, syntax_ptr)
                } else {
                    let callee = self.collect_expr_opt(e.expr());
                    let args = if let Some(arg_list) = e.arg_list() {
                        arg_list.args().filter_map(|e| self.maybe_collect_expr(e)).collect()
                    } else {
                        Box::default()
                    };
                    self.alloc_expr(
                        Expr::Call {
                            callee,
                            args,
                            is_assignee_expr: self.is_lowering_assignee_expr,
                        },
                        syntax_ptr,
                    )
                }
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
                                pat: self.collect_pat_top(arm.pat()),
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
            ast::Expr::ContinueExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.source_map.diagnostics.push(e);
                    None
                });
                self.alloc_expr(Expr::Continue { label }, syntax_ptr)
            }
            ast::Expr::BreakExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.source_map.diagnostics.push(e);
                    None
                });
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Break { expr, label }, syntax_ptr)
            }
            ast::Expr::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well
                let src = self.expander.in_file(syntax_ptr);
                self.source_map.expr_map.insert(src, inner);
                inner
            }
            ast::Expr::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Return { expr }, syntax_ptr)
            }
            ast::Expr::BecomeExpr(e) => {
                let expr =
                    e.expr().map(|e| self.collect_expr(e)).unwrap_or_else(|| self.missing_expr());
                self.alloc_expr(Expr::Become { expr }, syntax_ptr)
            }
            ast::Expr::YieldExpr(e) => {
                self.is_lowering_coroutine = true;
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
                            let src = self.expander.in_file(AstPtr::new(&field));
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
            ast::Expr::TryExpr(e) => self.collect_try_operator(syntax_ptr, e),
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
            ast::Expr::ClosureExpr(e) => self.with_label_rib(RibKind::Closure, |this| {
                let (result_expr_id, prev_binding_owner) =
                    this.initialize_binding_owner(syntax_ptr);
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    let num_params = pl.params().count();
                    args.reserve_exact(num_params);
                    arg_types.reserve_exact(num_params);
                    for param in pl.params() {
                        let pat = this.collect_pat_top(param.pat());
                        let type_ref =
                            param.ty().map(|it| Interned::new(TypeRef::from_ast(&this.ctx(), it)));
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let ret_type = e
                    .ret_type()
                    .and_then(|r| r.ty())
                    .map(|it| Interned::new(TypeRef::from_ast(&this.ctx(), it)));

                let prev_is_lowering_coroutine = mem::take(&mut this.is_lowering_coroutine);
                let prev_try_block_label = this.current_try_block_label.take();

                let body = this.collect_expr_opt(e.body());

                let closure_kind = if this.is_lowering_coroutine {
                    let movability = if e.static_token().is_some() {
                        Movability::Static
                    } else {
                        Movability::Movable
                    };
                    ClosureKind::Coroutine(movability)
                } else if e.async_token().is_some() {
                    ClosureKind::Async
                } else {
                    ClosureKind::Closure
                };
                let capture_by =
                    if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                this.is_lowering_coroutine = prev_is_lowering_coroutine;
                this.current_binding_owner = prev_binding_owner;
                this.current_try_block_label = prev_try_block_label;
                this.body.exprs[result_expr_id] = Expr::Closure {
                    args: args.into(),
                    arg_types: arg_types.into(),
                    ret_type,
                    body,
                    closure_kind,
                    capture_by,
                };
                result_expr_id
            }),
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
                let mut exprs: Vec<_> = e.fields().map(|expr| self.collect_expr(expr)).collect();
                // if there is a leading comma, the user is most likely to type out a leading expression
                // so we insert a missing expression at the beginning for IDE features
                if comma_follows_token(e.l_paren_token()) {
                    exprs.insert(0, self.missing_expr());
                }

                self.alloc_expr(
                    Expr::Tuple {
                        exprs: exprs.into_boxed_slice(),
                        is_assignee_expr: self.is_lowering_assignee_expr,
                    },
                    syntax_ptr,
                )
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
                        let repeat = self.with_label_rib(RibKind::Constant, |this| {
                            if let Some(repeat) = repeat {
                                let syntax_ptr = AstPtr::new(&repeat);
                                this.collect_as_a_binding_owner_bad(
                                    |this| this.collect_expr(repeat),
                                    syntax_ptr,
                                )
                            } else {
                                this.missing_expr()
                            }
                        });
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
                let is_assignee_expr = self.is_lowering_assignee_expr;
                self.alloc_expr(Expr::Index { base, index, is_assignee_expr }, syntax_ptr)
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
                        let src = self.expander.in_file(syntax_ptr);
                        self.source_map.expr_map.insert(src, id);
                        id
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::UnderscoreExpr(_) => self.alloc_expr(Expr::Underscore, syntax_ptr),
            ast::Expr::AsmExpr(e) => {
                let e = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::InlineAsm(InlineAsm { e }), syntax_ptr)
            }
            ast::Expr::OffsetOfExpr(e) => {
                let container = Interned::new(TypeRef::from_ast_opt(&self.ctx(), e.ty()));
                let fields = e.fields().map(|it| it.as_name()).collect();
                self.alloc_expr(Expr::OffsetOf(OffsetOf { container, fields }), syntax_ptr)
            }
            ast::Expr::FormatArgsExpr(f) => self.collect_format_args(f, syntax_ptr),
        })
    }

    fn initialize_binding_owner(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> (ExprId, Option<ExprId>) {
        let result_expr_id = self.alloc_expr(Expr::Missing, syntax_ptr);
        let prev_binding_owner = self.current_binding_owner.take();
        self.current_binding_owner = Some(result_expr_id);

        (result_expr_id, prev_binding_owner)
    }

    /// FIXME: This function is bad. It will produce a dangling `Missing` expr which wastes memory. Currently
    /// it is used only for const blocks and repeat expressions, which are also hacky and ideally should have
    /// their own body. Don't add more usage for this function so that we can remove this function after
    /// separating those bodies.
    fn collect_as_a_binding_owner_bad(
        &mut self,
        job: impl FnOnce(&mut ExprCollector<'_>) -> ExprId,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let (id, prev_owner) = self.initialize_binding_owner(syntax_ptr);
        let tmp = job(self);
        self.body.exprs[id] = mem::replace(&mut self.body.exprs[tmp], Expr::Missing);
        self.current_binding_owner = prev_owner;
        id
    }

    /// Desugar `try { <stmts>; <expr> }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(<expr>) }`,
    /// `try { <stmts>; }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(()) }`
    /// and save the `<new_label>` to use it as a break target for desugaring of the `?` operator.
    fn desugar_try_block(&mut self, e: BlockExpr) -> ExprId {
        let Some(try_from_output) = LangItem::TryTraitFromOutput.path(self.db, self.krate) else {
            return self.collect_block(e);
        };
        let label = self.alloc_label_desugared(Label { name: Name::generate_new_name() });
        let old_label = self.current_try_block_label.replace(label);

        let (btail, expr_id) = self.with_labeled_rib(label, |this| {
            let mut btail = None;
            let block = this.collect_block_(e, |id, statements, tail| {
                btail = tail;
                Expr::Block { id, statements, tail, label: Some(label) }
            });
            (btail, block)
        });

        let callee = self.alloc_expr_desugared(Expr::Path(try_from_output));
        let next_tail = match btail {
            Some(tail) => self.alloc_expr_desugared(Expr::Call {
                callee,
                args: Box::new([tail]),
                is_assignee_expr: false,
            }),
            None => {
                let unit = self.alloc_expr_desugared(Expr::Tuple {
                    exprs: Box::new([]),
                    is_assignee_expr: false,
                });
                self.alloc_expr_desugared(Expr::Call {
                    callee,
                    args: Box::new([unit]),
                    is_assignee_expr: false,
                })
            }
        };
        let Expr::Block { tail, .. } = &mut self.body.exprs[expr_id] else {
            unreachable!("block was lowered to non-block");
        };
        *tail = Some(next_tail);
        self.current_try_block_label = old_label;
        expr_id
    }

    /// Desugar `ast::WhileExpr` from: `[opt_ident]: while <cond> <body>` into:
    /// ```ignore (pseudo-rust)
    /// [opt_ident]: loop {
    ///   if <cond> {
    ///     <body>
    ///   }
    ///   else {
    ///     break;
    ///   }
    /// }
    /// ```
    /// FIXME: Rustc wraps the condition in a construct equivalent to `{ let _t = <cond>; _t }`
    /// to preserve drop semantics. We should probably do the same in future.
    fn collect_while_loop(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::WhileExpr) -> ExprId {
        let label = e.label().map(|label| self.collect_label(label));
        let body = self.collect_labelled_block_opt(label, e.loop_body());

        // Labels can also be used in the condition expression, like this:
        // ```
        // fn main() {
        //     let mut optional = Some(0);
        //     'my_label: while let Some(a) = match optional {
        //         None => break 'my_label,
        //         Some(val) => Some(val),
        //     } {
        //         println!("{}", a);
        //         optional = None;
        //     }
        // }
        // ```
        let condition = match label {
            Some(label) => {
                self.with_labeled_rib(label, |this| this.collect_expr_opt(e.condition()))
            }
            None => self.collect_expr_opt(e.condition()),
        };

        let break_expr = self.alloc_expr(Expr::Break { expr: None, label: None }, syntax_ptr);
        let if_expr = self.alloc_expr(
            Expr::If { condition, then_branch: body, else_branch: Some(break_expr) },
            syntax_ptr,
        );
        self.alloc_expr(Expr::Loop { body: if_expr, label }, syntax_ptr)
    }

    /// Desugar `ast::ForExpr` from: `[opt_ident]: for <pat> in <head> <body>` into:
    /// ```ignore (pseudo-rust)
    /// match IntoIterator::into_iter(<head>) {
    ///     mut iter => {
    ///         [opt_ident]: loop {
    ///             match Iterator::next(&mut iter) {
    ///                 None => break,
    ///                 Some(<pat>) => <body>,
    ///             };
    ///         }
    ///     }
    /// }
    /// ```
    fn collect_for_loop(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::ForExpr) -> ExprId {
        let Some((into_iter_fn, iter_next_fn, option_some, option_none)) = (|| {
            Some((
                LangItem::IntoIterIntoIter.path(self.db, self.krate)?,
                LangItem::IteratorNext.path(self.db, self.krate)?,
                LangItem::OptionSome.path(self.db, self.krate)?,
                LangItem::OptionNone.path(self.db, self.krate)?,
            ))
        })() else {
            // Some of the needed lang items are missing, so we can't desugar
            return self.alloc_expr(Expr::Missing, syntax_ptr);
        };
        let head = self.collect_expr_opt(e.iterable());
        let into_iter_fn_expr = self.alloc_expr(Expr::Path(into_iter_fn), syntax_ptr);
        let iterator = self.alloc_expr(
            Expr::Call {
                callee: into_iter_fn_expr,
                args: Box::new([head]),
                is_assignee_expr: false,
            },
            syntax_ptr,
        );
        let none_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::Path(Box::new(option_none))),
            guard: None,
            expr: self.alloc_expr(Expr::Break { expr: None, label: None }, syntax_ptr),
        };
        let some_pat = Pat::TupleStruct {
            path: Some(Box::new(option_some)),
            args: Box::new([self.collect_pat_top(e.pat())]),
            ellipsis: None,
        };
        let label = e.label().map(|label| self.collect_label(label));
        let some_arm = MatchArm {
            pat: self.alloc_pat_desugared(some_pat),
            guard: None,
            expr: self.with_opt_labeled_rib(label, |this| {
                this.collect_expr_opt(e.loop_body().map(|it| it.into()))
            }),
        };
        let iter_name = Name::generate_new_name();
        let iter_expr = self.alloc_expr(Expr::Path(Path::from(iter_name.clone())), syntax_ptr);
        let iter_expr_mut = self.alloc_expr(
            Expr::Ref { expr: iter_expr, rawness: Rawness::Ref, mutability: Mutability::Mut },
            syntax_ptr,
        );
        let iter_next_fn_expr = self.alloc_expr(Expr::Path(iter_next_fn), syntax_ptr);
        let iter_next_expr = self.alloc_expr(
            Expr::Call {
                callee: iter_next_fn_expr,
                args: Box::new([iter_expr_mut]),
                is_assignee_expr: false,
            },
            syntax_ptr,
        );
        let loop_inner = self.alloc_expr(
            Expr::Match { expr: iter_next_expr, arms: Box::new([none_arm, some_arm]) },
            syntax_ptr,
        );
        let loop_outer = self.alloc_expr(Expr::Loop { body: loop_inner, label }, syntax_ptr);
        let iter_binding = self.alloc_binding(iter_name, BindingAnnotation::Mutable);
        let iter_pat = self.alloc_pat_desugared(Pat::Bind { id: iter_binding, subpat: None });
        self.add_definition_to_binding(iter_binding, iter_pat);
        self.alloc_expr(
            Expr::Match {
                expr: iterator,
                arms: Box::new([MatchArm { pat: iter_pat, guard: None, expr: loop_outer }]),
            },
            syntax_ptr,
        )
    }

    /// Desugar `ast::TryExpr` from: `<expr>?` into:
    /// ```ignore (pseudo-rust)
    /// match Try::branch(<expr>) {
    ///     ControlFlow::Continue(val) => val,
    ///     ControlFlow::Break(residual) =>
    ///         // If there is an enclosing `try {...}`:
    ///         break 'catch_target Try::from_residual(residual),
    ///         // Otherwise:
    ///         return Try::from_residual(residual),
    /// }
    /// ```
    fn collect_try_operator(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::TryExpr) -> ExprId {
        let Some((try_branch, cf_continue, cf_break, try_from_residual)) = (|| {
            Some((
                LangItem::TryTraitBranch.path(self.db, self.krate)?,
                LangItem::ControlFlowContinue.path(self.db, self.krate)?,
                LangItem::ControlFlowBreak.path(self.db, self.krate)?,
                LangItem::TryTraitFromResidual.path(self.db, self.krate)?,
            ))
        })() else {
            // Some of the needed lang items are missing, so we can't desugar
            return self.alloc_expr(Expr::Missing, syntax_ptr);
        };
        let operand = self.collect_expr_opt(e.expr());
        let try_branch = self.alloc_expr(Expr::Path(try_branch), syntax_ptr);
        let expr = self.alloc_expr(
            Expr::Call { callee: try_branch, args: Box::new([operand]), is_assignee_expr: false },
            syntax_ptr,
        );
        let continue_name = Name::generate_new_name();
        let continue_binding =
            self.alloc_binding(continue_name.clone(), BindingAnnotation::Unannotated);
        let continue_bpat =
            self.alloc_pat_desugared(Pat::Bind { id: continue_binding, subpat: None });
        self.add_definition_to_binding(continue_binding, continue_bpat);
        let continue_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: Some(Box::new(cf_continue)),
                args: Box::new([continue_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: self.alloc_expr(Expr::Path(Path::from(continue_name)), syntax_ptr),
        };
        let break_name = Name::generate_new_name();
        let break_binding = self.alloc_binding(break_name.clone(), BindingAnnotation::Unannotated);
        let break_bpat = self.alloc_pat_desugared(Pat::Bind { id: break_binding, subpat: None });
        self.add_definition_to_binding(break_binding, break_bpat);
        let break_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: Some(Box::new(cf_break)),
                args: Box::new([break_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: {
                let it = self.alloc_expr(Expr::Path(Path::from(break_name)), syntax_ptr);
                let callee = self.alloc_expr(Expr::Path(try_from_residual), syntax_ptr);
                let result = self.alloc_expr(
                    Expr::Call { callee, args: Box::new([it]), is_assignee_expr: false },
                    syntax_ptr,
                );
                self.alloc_expr(
                    match self.current_try_block_label {
                        Some(label) => Expr::Break { expr: Some(result), label: Some(label) },
                        None => Expr::Return { expr: Some(result) },
                    },
                    syntax_ptr,
                )
            },
        };
        let arms = Box::new([continue_arm, break_arm]);
        self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
    }

    fn collect_macro_call<T, U>(
        &mut self,
        mcall: ast::MacroCall,
        syntax_ptr: AstPtr<ast::MacroCall>,
        record_diagnostics: bool,
        collector: impl FnOnce(&mut Self, Option<T>) -> U,
    ) -> U
    where
        T: ast::AstNode,
    {
        // File containing the macro call. Expansion errors will be attached here.
        let outer_file = self.expander.current_file_id();

        let macro_call_ptr = self.expander.in_file(syntax_ptr);
        let module = self.expander.module.local_id;

        let res = match self.def_map.modules[module]
            .scope
            .macro_invoc(InFile::new(outer_file, self.ast_id_map.ast_id_for_ptr(syntax_ptr)))
        {
            // fast path, macro call is in a block module
            Some(call) => Ok(self.expander.enter_expand_id(self.db, call)),
            None => self.expander.enter_expand(self.db, mcall, |path| {
                self.def_map
                    .resolve_path(
                        self.db,
                        module,
                        &path,
                        crate::item_scope::BuiltinShadowMode::Other,
                        Some(MacroSubNs::Bang),
                    )
                    .0
                    .take_macros()
            }),
        };

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
                self.source_map.expansions.insert(macro_call_ptr, self.expander.current_file_id());
                let prev_ast_id_map = mem::replace(
                    &mut self.ast_id_map,
                    self.db.ast_id_map(self.expander.current_file_id()),
                );

                if record_diagnostics {
                    // FIXME: Report parse errors here
                }

                let id = collector(self, Some(expansion.tree()));
                self.ast_id_map = prev_ast_id_map;
                self.expander.exit(mark);
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
                let src = self.expander.in_file(syntax_ptr);
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
                let pat = self.collect_pat_top(stmt.pat());
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
            ast::Stmt::Item(_item) => statements.push(Statement::Item),
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
        mk_block: impl FnOnce(Option<BlockId>, Box<[Statement]>, Option<ExprId>) -> Expr,
    ) -> ExprId {
        let block_has_items = {
            let statement_has_item = block.statements().any(|stmt| match stmt {
                ast::Stmt::Item(_) => true,
                // Macro calls can be both items and expressions. The syntax library always treats
                // them as expressions here, so we undo that.
                ast::Stmt::ExprStmt(es) => matches!(es.expr(), Some(ast::Expr::MacroExpr(_))),
                _ => false,
            });
            statement_has_item
                || matches!(block.tail_expr(), Some(ast::Expr::MacroExpr(_)))
                || (block.may_carry_attributes() && block.attrs().next().is_some())
        };

        let block_id = if block_has_items {
            let file_local_id = self.ast_id_map.ast_id(&block);
            let ast_id = self.expander.in_file(file_local_id);
            Some(self.db.intern_block(BlockLoc { ast_id, module: self.expander.module }))
        } else {
            None
        };

        let (module, def_map) =
            match block_id.map(|block_id| (self.db.block_def_map(block_id), block_id)) {
                Some((def_map, block_id)) => {
                    self.body.block_scopes.push(block_id);
                    (def_map.module_id(DefMap::ROOT), def_map)
                }
                None => (self.expander.module, self.def_map.clone()),
            };
        let prev_def_map = mem::replace(&mut self.def_map, def_map);
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

        self.def_map = prev_def_map;
        self.expander.module = prev_local_module;
        expr_id
    }

    fn collect_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> ExprId {
        match expr {
            Some(block) => self.collect_block(block),
            None => self.missing_expr(),
        }
    }

    fn collect_labelled_block_opt(
        &mut self,
        label: Option<LabelId>,
        expr: Option<ast::BlockExpr>,
    ) -> ExprId {
        match label {
            Some(label) => self.with_labeled_rib(label, |this| this.collect_block_opt(expr)),
            None => self.collect_block_opt(expr),
        }
    }

    // region: patterns

    fn collect_pat_top(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat, &mut BindingList::default()),
            None => self.missing_pat(),
        }
    }

    fn collect_pat(&mut self, pat: ast::Pat, binding_list: &mut BindingList) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);

                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat(subpat, binding_list));

                let is_simple_ident_pat =
                    annotation == BindingAnnotation::Unannotated && subpat.is_none();
                let (binding, pattern) = if is_simple_ident_pat {
                    // This could also be a single-segment path pattern. To
                    // decide that, we need to try resolving the name.
                    let (resolved, _) = self.def_map.resolve_path(
                        self.db,
                        self.expander.module.local_id,
                        &name.clone().into(),
                        BuiltinShadowMode::Other,
                        None,
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

                let ptr = AstPtr::new(&Either::Left(pat));
                let pat = self.alloc_pat(pattern, ptr);
                if let Some(binding_id) = binding {
                    self.add_definition_to_binding(binding_id, pat);
                }
                return pat;
            }
            ast::Pat::TupleStructPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat(), binding_list);
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => 'b: {
                let prev_is_used = mem::take(&mut binding_list.is_used);
                let prev_reject_new = mem::take(&mut binding_list.reject_new);
                let mut pats = Vec::with_capacity(p.pats().count());
                let mut it = p.pats();
                let Some(first) = it.next() else {
                    break 'b Pat::Or(Box::new([]));
                };
                pats.push(self.collect_pat(first, binding_list));
                binding_list.reject_new = true;
                for rest in it {
                    for (_, it) in binding_list.is_used.iter_mut() {
                        *it = false;
                    }
                    pats.push(self.collect_pat(rest, binding_list));
                    for (&id, &is_used) in binding_list.is_used.iter() {
                        if !is_used {
                            self.body.bindings[id].problems =
                                Some(BindingProblems::NotBoundAcrossAll);
                        }
                    }
                }
                binding_list.reject_new = prev_reject_new;
                let current_is_used = mem::replace(&mut binding_list.is_used, prev_is_used);
                for (id, _) in current_is_used.into_iter() {
                    binding_list.check_is_used(self, id);
                }
                Pat::Or(pats.into())
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt(p.pat(), binding_list),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let record_pat_field_list =
                    &p.record_pat_field_list().expect("every struct should have a field list");
                let args = record_pat_field_list
                    .fields()
                    .filter_map(|f| {
                        self.check_cfg(&f)?;
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat(ast_pat, binding_list);
                        let name = f.field_name()?.as_name();
                        let src = self.expander.in_file(AstPtr::new(&f));
                        self.source_map.pat_field_map_back.insert(pat, src);
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = record_pat_field_list.rest_pat().is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                // FIXME properly handle `RestPat`
                Pat::Slice {
                    prefix: prefix.into_iter().map(|p| self.collect_pat(p, binding_list)).collect(),
                    slice: slice.map(|p| self.collect_pat(p, binding_list)),
                    suffix: suffix.into_iter().map(|p| self.collect_pat(p, binding_list)).collect(),
                }
            }
            #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5676
            ast::Pat::LiteralPat(lit) => 'b: {
                let Some((hir_lit, ast_lit)) = pat_literal_to_hir(lit) else { break 'b Pat::Missing };
                let expr = Expr::Literal(hir_lit);
                let expr_ptr = AstPtr::new(&ast::Expr::Literal(ast_lit));
                let expr_id = self.alloc_expr(expr, expr_ptr);
                Pat::Lit(expr_id)
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
                let inner = self.collect_pat_opt(boxpat.pat(), binding_list);
                Pat::Box { inner }
            }
            ast::Pat::ConstBlockPat(const_block_pat) => {
                if let Some(block) = const_block_pat.block_expr() {
                    let expr_id = self.with_label_rib(RibKind::Constant, |this| {
                        let syntax_ptr = AstPtr::new(&block.clone().into());
                        this.collect_as_a_binding_owner_bad(
                            |this| this.collect_block(block),
                            syntax_ptr,
                        )
                    });
                    Pat::ConstBlock(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::MacroPat(mac) => match mac.macro_call() {
                Some(call) => {
                    let macro_ptr = AstPtr::new(&call);
                    let src = self.expander.in_file(AstPtr::new(&Either::Left(pat)));
                    let pat =
                        self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                            this.collect_pat_opt(expanded_pat, binding_list)
                        });
                    self.source_map.pat_map.insert(src, pat);
                    return pat;
                }
                None => Pat::Missing,
            },
            // FIXME: implement in a way that also builds source map and calculates assoc resolutions in type inference.
            ast::Pat::RangePat(p) => {
                let mut range_part_lower = |p: Option<ast::Pat>| {
                    p.and_then(|it| match &it {
                        ast::Pat::LiteralPat(it) => {
                            Some(Box::new(LiteralOrConst::Literal(pat_literal_to_hir(it)?.0)))
                        }
                        ast::Pat::IdentPat(p) => {
                            let name =
                                p.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                            Some(Box::new(LiteralOrConst::Const(name.into())))
                        }
                        ast::Pat::PathPat(p) => p
                            .path()
                            .and_then(|path| self.expander.parse_path(self.db, path))
                            .map(LiteralOrConst::Const)
                            .map(Box::new),
                        _ => None,
                    })
                };
                let start = range_part_lower(p.start());
                let end = range_part_lower(p.end());
                Pat::Range { start, end }
            }
        };
        let ptr = AstPtr::new(&Either::Left(pat));
        self.alloc_pat(pattern, ptr)
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>, binding_list: &mut BindingList) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat, binding_list),
            None => self.missing_pat(),
        }
    }

    fn collect_tuple_pat(
        &mut self,
        args: AstChildren<ast::Pat>,
        has_leading_comma: bool,
        binding_list: &mut BindingList,
    ) -> (Box<[PatId]>, Option<usize>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis = args.clone().position(|p| matches!(p, ast::Pat::RestPat(_)));
        // We want to skip the `..` pattern here, since we account for it above.
        let mut args: Vec<_> = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.collect_pat(p, binding_list))
            .collect();
        // if there is a leading comma, the user is most likely to type out a leading pattern
        // so we insert a missing pattern at the beginning for IDE features
        if has_leading_comma {
            args.insert(0, self.missing_pat());
        }

        (args.into_boxed_slice(), ellipsis)
    }

    // endregion: patterns

    /// Returns `None` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `Some(())` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::HasAttrs) -> Option<()> {
        match self.expander.parse_attrs(self.db, owner).cfg() {
            Some(cfg) => {
                if self.expander.cfg_options().check(&cfg) != Some(false) {
                    return Some(());
                }

                self.source_map.diagnostics.push(BodyDiagnostic::InactiveCode {
                    node: self.expander.in_file(SyntaxNodePtr::new(owner.syntax())),
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

    // region: labels

    fn collect_label(&mut self, ast_label: ast::Label) -> LabelId {
        let label = Label {
            name: ast_label.lifetime().as_ref().map_or_else(Name::missing, Name::new_lifetime),
        };
        self.alloc_label(label, AstPtr::new(&ast_label))
    }

    fn resolve_label(
        &self,
        lifetime: Option<ast::Lifetime>,
    ) -> Result<Option<LabelId>, BodyDiagnostic> {
        let Some(lifetime) = lifetime else { return Ok(None) };
        let name = Name::new_lifetime(&lifetime);

        for (rib_idx, rib) in self.label_ribs.iter().enumerate().rev() {
            if let Some((label_name, id)) = &rib.label {
                if *label_name == name {
                    return if self.is_label_valid_from_rib(rib_idx) {
                        Ok(Some(*id))
                    } else {
                        Err(BodyDiagnostic::UnreachableLabel {
                            name,
                            node: self.expander.in_file(AstPtr::new(&lifetime)),
                        })
                    };
                }
            }
        }

        Err(BodyDiagnostic::UndeclaredLabel {
            name,
            node: self.expander.in_file(AstPtr::new(&lifetime)),
        })
    }

    fn is_label_valid_from_rib(&self, rib_index: usize) -> bool {
        !self.label_ribs[rib_index + 1..].iter().any(|rib| rib.kind.is_label_barrier())
    }

    fn with_label_rib<T>(&mut self, kind: RibKind, f: impl FnOnce(&mut Self) -> T) -> T {
        self.label_ribs.push(LabelRib::new(kind));
        let res = f(self);
        self.label_ribs.pop();
        res
    }

    fn with_labeled_rib<T>(&mut self, label: LabelId, f: impl FnOnce(&mut Self) -> T) -> T {
        self.label_ribs.push(LabelRib::new_normal((self.body[label].name.clone(), label)));
        let res = f(self);
        self.label_ribs.pop();
        res
    }

    fn with_opt_labeled_rib<T>(
        &mut self,
        label: Option<LabelId>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match label {
            None => f(self),
            Some(label) => self.with_labeled_rib(label, f),
        }
    }
    // endregion: labels

    // region: format
    fn expand_macros_to_string(&mut self, expr: ast::Expr) -> Option<(ast::String, bool)> {
        let m = match expr {
            ast::Expr::MacroExpr(m) => m,
            ast::Expr::Literal(l) => {
                return match l.kind() {
                    ast::LiteralKind::String(s) => Some((s, true)),
                    _ => None,
                }
            }
            _ => return None,
        };
        let e = m.macro_call()?;
        let macro_ptr = AstPtr::new(&e);
        let (exp, _) = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
            expansion.and_then(|it| this.expand_macros_to_string(it))
        })?;
        Some((exp, false))
    }

    fn collect_format_args(
        &mut self,
        f: ast::FormatArgsExpr,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let mut args = FormatArgumentsCollector::new();
        f.args().for_each(|arg| {
            args.add(FormatArgument {
                kind: match arg.name() {
                    Some(name) => FormatArgumentKind::Named(name.as_name()),
                    None => FormatArgumentKind::Normal,
                },
                expr: self.collect_expr_opt(arg.expr()),
            });
        });
        let template = f.template();
        let fmt_snippet = template.as_ref().map(ToString::to_string);
        let mut mappings = vec![];
        let fmt = match template.and_then(|it| self.expand_macros_to_string(it)) {
            Some((s, is_direct_literal)) => format_args::parse(
                &s,
                fmt_snippet,
                args,
                is_direct_literal,
                |name| self.alloc_expr_desugared(Expr::Path(Path::from(name))),
                |name, span| {
                    if let Some(span) = span {
                        mappings.push((span, name))
                    }
                },
            ),
            None => FormatArgs {
                template: Default::default(),
                arguments: args.finish(),
                orphans: Default::default(),
            },
        };

        // Create a list of all _unique_ (argument, format trait) combinations.
        // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
        let mut argmap = FxIndexSet::default();
        for piece in fmt.template.iter() {
            let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
            if let Ok(index) = placeholder.argument.index {
                argmap.insert((index, ArgumentType::Format(placeholder.format_trait)));
            }
        }

        let lit_pieces =
            fmt.template
                .iter()
                .enumerate()
                .filter_map(|(i, piece)| {
                    match piece {
                        FormatArgsPiece::Literal(s) => Some(
                            self.alloc_expr_desugared(Expr::Literal(Literal::String(s.clone()))),
                        ),
                        &FormatArgsPiece::Placeholder(_) => {
                            // Inject empty string before placeholders when not already preceded by a literal piece.
                            if i == 0
                                || matches!(fmt.template[i - 1], FormatArgsPiece::Placeholder(_))
                            {
                                Some(self.alloc_expr_desugared(Expr::Literal(Literal::String(
                                    "".into(),
                                ))))
                            } else {
                                None
                            }
                        }
                    }
                })
                .collect();
        let lit_pieces = self.alloc_expr_desugared(Expr::Array(Array::ElementList {
            elements: lit_pieces,
            is_assignee_expr: false,
        }));
        let lit_pieces = self.alloc_expr_desugared(Expr::Ref {
            expr: lit_pieces,
            rawness: Rawness::Ref,
            mutability: Mutability::Shared,
        });
        let format_options = {
            // Generate:
            //     &[format_spec_0, format_spec_1, format_spec_2]
            let elements = fmt
                .template
                .iter()
                .filter_map(|piece| {
                    let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
                    Some(self.make_format_spec(placeholder, &mut argmap))
                })
                .collect();
            let array = self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                elements,
                is_assignee_expr: false,
            }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };
        let arguments = &*fmt.arguments.arguments;

        let args = if arguments.is_empty() {
            let expr = self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                elements: Box::default(),
                is_assignee_expr: false,
            }));
            self.alloc_expr_desugared(Expr::Ref {
                expr,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        } else {
            // Generate:
            //     &match (&arg0, &arg1, &…) {
            //         args => [
            //             <core::fmt::Argument>::new_display(args.0),
            //             <core::fmt::Argument>::new_lower_hex(args.1),
            //             <core::fmt::Argument>::new_debug(args.0),
            //             …
            //         ]
            //     }
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let arg = self.alloc_expr_desugared(Expr::Ref {
                        expr: arguments[arg_index].expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let array = self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                elements: args,
                is_assignee_expr: false,
            }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };

        // Generate:
        //     <core::fmt::Arguments>::new_v1_formatted(
        //         lit_pieces,
        //         args,
        //         format_options,
        //         unsafe { ::core::fmt::UnsafeArg::new() }
        //     )

        let Some(new_v1_formatted) =
            LangItem::FormatArguments.ty_rel_path(self.db, self.krate, name![new_v1_formatted])
        else {
            return self.missing_expr();
        };
        let Some(unsafe_arg_new) =
            LangItem::FormatUnsafeArg.ty_rel_path(self.db, self.krate, name![new])
        else {
            return self.missing_expr();
        };
        let new_v1_formatted = self.alloc_expr_desugared(Expr::Path(new_v1_formatted));

        let unsafe_arg_new = self.alloc_expr_desugared(Expr::Path(unsafe_arg_new));
        let unsafe_arg_new = self.alloc_expr_desugared(Expr::Call {
            callee: unsafe_arg_new,
            args: Box::default(),
            is_assignee_expr: false,
        });
        let unsafe_arg_new = self.alloc_expr_desugared(Expr::Unsafe {
            id: None,
            // We collect the unused expressions here so that we still infer them instead of
            // dropping them out of the expression tree
            statements: fmt
                .orphans
                .into_iter()
                .map(|expr| Statement::Expr { expr, has_semi: true })
                .collect(),
            tail: Some(unsafe_arg_new),
        });

        let idx = self.alloc_expr(
            Expr::Call {
                callee: new_v1_formatted,
                args: Box::new([lit_pieces, args, format_options, unsafe_arg_new]),
                is_assignee_expr: false,
            },
            syntax_ptr,
        );
        self.source_map.format_args_template_map.insert(idx, mappings);
        idx
    }

    /// Generate a hir expression for a format_args placeholder specification.
    ///
    /// Generates
    ///
    /// ```text
    ///     <core::fmt::rt::Placeholder::new(
    ///         …usize, // position
    ///         '…', // fill
    ///         <core::fmt::rt::Alignment>::…, // alignment
    ///         …u32, // flags
    ///         <core::fmt::rt::Count::…>, // width
    ///         <core::fmt::rt::Count::…>, // precision
    ///     )
    /// ```
    fn make_format_spec(
        &mut self,
        placeholder: &FormatPlaceholder,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        let position = match placeholder.argument.index {
            Ok(arg_index) => {
                let (i, _) =
                    argmap.insert_full((arg_index, ArgumentType::Format(placeholder.format_trait)));
                self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                    i as u128,
                    Some(BuiltinUint::Usize),
                )))
            }
            Err(_) => self.missing_expr(),
        };
        let &FormatOptions {
            ref width,
            ref precision,
            alignment,
            fill,
            sign,
            alternate,
            zero_pad,
            debug_hex,
        } = &placeholder.format_options;
        let fill = self.alloc_expr_desugared(Expr::Literal(Literal::Char(fill.unwrap_or(' '))));

        let align = {
            let align = LangItem::FormatAlignment.ty_rel_path(
                self.db,
                self.krate,
                match alignment {
                    Some(FormatAlignment::Left) => name![Left],
                    Some(FormatAlignment::Right) => name![Right],
                    Some(FormatAlignment::Center) => name![Center],
                    None => name![Unknown],
                },
            );
            match align {
                Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                None => self.missing_expr(),
            }
        };
        // This needs to match `Flag` in library/core/src/fmt/rt.rs.
        let flags: u32 = ((sign == Some(FormatSign::Plus)) as u32)
            | ((sign == Some(FormatSign::Minus)) as u32) << 1
            | (alternate as u32) << 2
            | (zero_pad as u32) << 3
            | ((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 4
            | ((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 5;
        let flags = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
            flags as u128,
            Some(BuiltinUint::U32),
        )));
        let precision = self.make_count(precision, argmap);
        let width = self.make_count(width, argmap);

        let format_placeholder_new = {
            let format_placeholder_new =
                LangItem::FormatPlaceholder.ty_rel_path(self.db, self.krate, name![new]);
            match format_placeholder_new {
                Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                None => self.missing_expr(),
            }
        };

        self.alloc_expr_desugared(Expr::Call {
            callee: format_placeholder_new,
            args: Box::new([position, fill, align, flags, precision, width]),
            is_assignee_expr: false,
        })
    }

    /// Generate a hir expression for a format_args Count.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Is(…)
    /// ```
    ///
    /// or
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Param(…)
    /// ```
    ///
    /// or
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Implied
    /// ```
    fn make_count(
        &mut self,
        count: &Option<FormatCount>,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        match count {
            Some(FormatCount::Literal(n)) => {
                match LangItem::FormatCount.ty_rel_path(self.db, self.krate, name![Is]) {
                    Some(count_is) => {
                        let count_is = self.alloc_expr_desugared(Expr::Path(count_is));
                        let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                            *n as u128,
                            Some(BuiltinUint::Usize),
                        )));
                        self.alloc_expr_desugared(Expr::Call {
                            callee: count_is,
                            args: Box::new([args]),
                            is_assignee_expr: false,
                        })
                    }
                    None => self.missing_expr(),
                }
            }
            Some(FormatCount::Argument(arg)) => {
                if let Ok(arg_index) = arg.index {
                    let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));

                    match LangItem::FormatCount.ty_rel_path(self.db, self.krate, name![Param]) {
                        Some(count_param) => {
                            let count_param = self.alloc_expr_desugared(Expr::Path(count_param));
                            let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                                i as u128,
                                Some(BuiltinUint::Usize),
                            )));
                            self.alloc_expr_desugared(Expr::Call {
                                callee: count_param,
                                args: Box::new([args]),
                                is_assignee_expr: false,
                            })
                        }
                        None => self.missing_expr(),
                    }
                } else {
                    self.missing_expr()
                }
            }
            None => match LangItem::FormatCount.ty_rel_path(self.db, self.krate, name![Implied]) {
                Some(count_param) => self.alloc_expr_desugared(Expr::Path(count_param)),
                None => self.missing_expr(),
            },
        }
    }

    /// Generate a hir expression representing an argument to a format_args invocation.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::Argument>::new_…(arg)
    /// ```
    fn make_argument(&mut self, arg: ExprId, ty: ArgumentType) -> ExprId {
        use ArgumentType::*;
        use FormatTrait::*;
        match LangItem::FormatArgument.ty_rel_path(
            self.db,
            self.krate,
            match ty {
                Format(Display) => name![new_display],
                Format(Debug) => name![new_debug],
                Format(LowerExp) => name![new_lower_exp],
                Format(UpperExp) => name![new_upper_exp],
                Format(Octal) => name![new_octal],
                Format(Pointer) => name![new_pointer],
                Format(Binary) => name![new_binary],
                Format(LowerHex) => name![new_lower_hex],
                Format(UpperHex) => name![new_upper_hex],
                Usize => name![from_usize],
            },
        ) {
            Some(new_fn) => {
                let new_fn = self.alloc_expr_desugared(Expr::Path(new_fn));
                self.alloc_expr_desugared(Expr::Call {
                    callee: new_fn,
                    args: Box::new([arg]),
                    is_assignee_expr: false,
                })
            }
            None => self.missing_expr(),
        }
    }
    // endregion: format
}

fn pat_literal_to_hir(lit: &ast::LiteralPat) -> Option<(Literal, ast::Literal)> {
    let ast_lit = lit.literal()?;
    let mut hir_lit: Literal = ast_lit.kind().into();
    if lit.minus_token().is_some() {
        hir_lit = hir_lit.negate()?;
    }
    Some((hir_lit, ast_lit))
}

impl ExprCollector<'_> {
    fn alloc_expr(&mut self, expr: Expr, ptr: ExprPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.body.exprs.alloc(expr);
        self.source_map.expr_map_back.insert(id, src);
        self.source_map.expr_map.insert(src, id);
        id
    }
    // FIXME: desugared exprs don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.body.exprs.alloc(expr)
    }
    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }

    fn alloc_binding(&mut self, name: Name, mode: BindingAnnotation) -> BindingId {
        let binding = self.body.bindings.alloc(Binding {
            name,
            mode,
            definitions: SmallVec::new(),
            problems: None,
        });
        if let Some(owner) = self.current_binding_owner {
            self.body.binding_owners.insert(binding, owner);
        }
        binding
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.in_file(ptr);
        let id = self.body.pats.alloc(pat);
        self.source_map.pat_map_back.insert(id, src);
        self.source_map.pat_map.insert(src, id);
        id
    }
    // FIXME: desugared pats don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_pat_desugared(&mut self, pat: Pat) -> PatId {
        self.body.pats.alloc(pat)
    }
    fn missing_pat(&mut self) -> PatId {
        self.body.pats.alloc(Pat::Missing)
    }

    fn alloc_label(&mut self, label: Label, ptr: LabelPtr) -> LabelId {
        let src = self.expander.in_file(ptr);
        let id = self.body.labels.alloc(label);
        self.source_map.label_map_back.insert(id, src);
        self.source_map.label_map.insert(src, id);
        id
    }
    // FIXME: desugared labels don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_label_desugared(&mut self, label: Label) -> LabelId {
        self.body.labels.alloc(label)
    }
}

fn comma_follows_token(t: Option<syntax::SyntaxToken>) -> bool {
    (|| syntax::algo::skip_trivia_token(t?.next_token()?, syntax::Direction::Next))()
        .map_or(false, |it| it.kind() == syntax::T![,])
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
}
