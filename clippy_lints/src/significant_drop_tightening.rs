use clippy_utils::{
    diagnostics::span_lint_and_then,
    expr_or_init, get_attr, path_to_local,
    source::{indent_of, snippet},
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir,
    intravisit::{walk_expr, Visitor},
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{subst::GenericArgKind, Ty, TypeAndMut};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::Ident, Span, DUMMY_SP};
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Searches for elements marked with `#[clippy::has_significant_drop]` that could be early
    /// dropped but are in fact dropped at the end of their scopes. In other words, enforces the
    /// "tightening" of their possible lifetimes.
    ///
    /// ### Why is this bad?
    ///
    /// Elements marked with `#[clippy::has_significant_drop]` are generally synchronizing
    /// primitives that manage shared resources, as such, it is desired to release them as soon as
    /// possible to avoid unnecessary resource contention.
    ///
    /// ### Example
    ///
    /// ```rust,ignore
    /// fn main() {
    ///   let lock = some_sync_resource.lock();
    ///   let owned_rslt = lock.do_stuff_with_resource();
    ///   // Only `owned_rslt` is needed but `lock` is still held.
    ///   do_heavy_computation_that_takes_time(owned_rslt);
    /// }
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust,ignore
    /// fn main() {
    ///     let owned_rslt = some_sync_resource.lock().do_stuff_with_resource();
    ///     do_heavy_computation_that_takes_time(owned_rslt);
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub SIGNIFICANT_DROP_TIGHTENING,
    nursery,
    "Searches for elements marked with `#[clippy::has_significant_drop]` that could be early dropped but are in fact dropped at the end of their scopes"
}

impl_lint_pass!(SignificantDropTightening<'_> => [SIGNIFICANT_DROP_TIGHTENING]);

#[derive(Default)]
pub struct SignificantDropTightening<'tcx> {
    apas: FxIndexMap<hir::HirId, AuxParamsAttr>,
    /// Auxiliary structure used to avoid having to verify the same type multiple times.
    seen_types: FxHashSet<Ty<'tcx>>,
    type_cache: FxHashMap<Ty<'tcx>, bool>,
}

impl<'tcx> LateLintPass<'tcx> for SignificantDropTightening<'tcx> {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: hir::intravisit::FnKind<'_>,
        _: &hir::FnDecl<'_>,
        body: &'tcx hir::Body<'_>,
        _: Span,
        _: hir::def_id::LocalDefId,
    ) {
        self.apas.clear();
        let initial_dummy_stmt = dummy_stmt_expr(body.value);
        let mut ap = AuxParams::new(&mut self.apas, &initial_dummy_stmt);
        StmtsChecker::new(&mut ap, cx, &mut self.seen_types, &mut self.type_cache).visit_body(body);
        for apa in ap.apas.values() {
            if apa.counter <= 1 || !apa.has_expensive_expr_after_last_attr {
                continue;
            }
            span_lint_and_then(
                cx,
                SIGNIFICANT_DROP_TIGHTENING,
                apa.first_bind_ident.span,
                "temporary with significant `Drop` can be early dropped",
                |diag| {
                    match apa.counter {
                        0 | 1 => {},
                        2 => {
                            let indent = " ".repeat(indent_of(cx, apa.last_stmt_span).unwrap_or(0));
                            let init_method = snippet(cx, apa.first_method_span, "..");
                            let usage_method = snippet(cx, apa.last_method_span, "..");
                            let stmt = if apa.last_bind_ident == Ident::empty() {
                                format!("\n{indent}{init_method}.{usage_method};")
                            } else {
                                format!(
                                    "\n{indent}let {} = {init_method}.{usage_method};",
                                    snippet(cx, apa.last_bind_ident.span, ".."),
                                )
                            };
                            diag.span_suggestion_verbose(
                                apa.first_stmt_span,
                                "merge the temporary construction with its single usage",
                                stmt,
                                Applicability::MaybeIncorrect,
                            );
                            diag.span_suggestion(
                                apa.last_stmt_span,
                                "remove separated single usage",
                                "",
                                Applicability::MaybeIncorrect,
                            );
                        },
                        _ => {
                            diag.span_suggestion(
                                apa.last_stmt_span.shrink_to_hi(),
                                "drop the temporary after the end of its last usage",
                                format!(
                                    "\n{}drop({});",
                                    " ".repeat(indent_of(cx, apa.last_stmt_span).unwrap_or(0)),
                                    apa.first_bind_ident
                                ),
                                Applicability::MaybeIncorrect,
                            );
                        },
                    }
                    diag.note("this might lead to unnecessary resource contention");
                    diag.span_label(
                        apa.first_block_span,
                        format!(
                            "temporary `{}` is currently being dropped at the end of its contained scope",
                            apa.first_bind_ident
                        ),
                    );
                },
            );
        }
    }
}

/// Checks the existence of the `#[has_significant_drop]` attribute.
struct AttrChecker<'cx, 'others, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    seen_types: &'others mut FxHashSet<Ty<'tcx>>,
    type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
}

impl<'cx, 'others, 'tcx> AttrChecker<'cx, 'others, 'tcx> {
    pub(crate) fn new(
        cx: &'cx LateContext<'tcx>,
        seen_types: &'others mut FxHashSet<Ty<'tcx>>,
        type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
    ) -> Self {
        seen_types.clear();
        Self {
            cx,
            seen_types,
            type_cache,
        }
    }

    fn has_sig_drop_attr(&mut self, ty: Ty<'tcx>) -> bool {
        // The borrow checker prevents us from using something fancier like or_insert_with.
        if let Some(ty) = self.type_cache.get(&ty) {
            return *ty;
        }
        let value = self.has_sig_drop_attr_uncached(ty);
        self.type_cache.insert(ty, value);
        value
    }

    fn has_sig_drop_attr_uncached(&mut self, ty: Ty<'tcx>) -> bool {
        if let Some(adt) = ty.ty_adt_def() {
            let mut iter = get_attr(
                self.cx.sess(),
                self.cx.tcx.get_attrs_unchecked(adt.did()),
                "has_significant_drop",
            );
            if iter.next().is_some() {
                return true;
            }
        }
        match ty.kind() {
            rustc_middle::ty::Adt(a, b) => {
                for f in a.all_fields() {
                    let ty = f.ty(self.cx.tcx, b);
                    if !self.has_seen_ty(ty) && self.has_sig_drop_attr(ty) {
                        return true;
                    }
                }
                for generic_arg in *b {
                    if let GenericArgKind::Type(ty) = generic_arg.unpack() {
                        if self.has_sig_drop_attr(ty) {
                            return true;
                        }
                    }
                }
                false
            },
            rustc_middle::ty::Array(ty, _)
            | rustc_middle::ty::RawPtr(TypeAndMut { ty, .. })
            | rustc_middle::ty::Ref(_, ty, _)
            | rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr(*ty),
            _ => false,
        }
    }

    fn has_seen_ty(&mut self, ty: Ty<'tcx>) -> bool {
        !self.seen_types.insert(ty)
    }
}

struct StmtsChecker<'ap, 'lc, 'others, 'stmt, 'tcx> {
    ap: &'ap mut AuxParams<'others, 'stmt, 'tcx>,
    cx: &'lc LateContext<'tcx>,
    seen_types: &'others mut FxHashSet<Ty<'tcx>>,
    type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
}

impl<'ap, 'lc, 'others, 'stmt, 'tcx> StmtsChecker<'ap, 'lc, 'others, 'stmt, 'tcx> {
    fn new(
        ap: &'ap mut AuxParams<'others, 'stmt, 'tcx>,
        cx: &'lc LateContext<'tcx>,
        seen_types: &'others mut FxHashSet<Ty<'tcx>>,
        type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
    ) -> Self {
        Self {
            ap,
            cx,
            seen_types,
            type_cache,
        }
    }

    fn manage_has_expensive_expr_after_last_attr(&mut self) {
        let has_expensive_stmt = match self.ap.curr_stmt.kind {
            hir::StmtKind::Expr(expr) if !is_expensive_expr(expr) => false,
            hir::StmtKind::Local(local) if let Some(expr) = local.init
                && let hir::ExprKind::Path(_) = expr.kind => false,
            _ => true
        };
        if has_expensive_stmt {
            for apa in self.ap.apas.values_mut() {
                let last_stmt_is_not_dummy = apa.last_stmt_span != DUMMY_SP;
                let last_stmt_is_not_curr = self.ap.curr_stmt.span != apa.last_stmt_span;
                let block_equals_curr = self.ap.curr_block_hir_id == apa.first_block_hir_id;
                let block_is_ancestor = self
                    .cx
                    .tcx
                    .hir()
                    .parent_iter(self.ap.curr_block_hir_id)
                    .any(|(id, _)| id == apa.first_block_hir_id);
                if last_stmt_is_not_dummy && last_stmt_is_not_curr && (block_equals_curr || block_is_ancestor) {
                    apa.has_expensive_expr_after_last_attr = true;
                }
            }
        }
    }
}

impl<'ap, 'lc, 'others, 'stmt, 'tcx> Visitor<'tcx> for StmtsChecker<'ap, 'lc, 'others, 'stmt, 'tcx> {
    fn visit_block(&mut self, block: &'tcx hir::Block<'tcx>) {
        self.ap.curr_block_hir_id = block.hir_id;
        self.ap.curr_block_span = block.span;
        for stmt in block.stmts {
            self.ap.curr_stmt = Cow::Borrowed(stmt);
            self.visit_stmt(stmt);
            self.ap.curr_block_hir_id = block.hir_id;
            self.ap.curr_block_span = block.span;
            self.manage_has_expensive_expr_after_last_attr();
        }
        if let Some(expr) = block.expr {
            self.ap.curr_stmt = Cow::Owned(dummy_stmt_expr(expr));
            self.visit_expr(expr);
            self.ap.curr_block_hir_id = block.hir_id;
            self.ap.curr_block_span = block.span;
            self.manage_has_expensive_expr_after_last_attr();
        }
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let modify_apa_params = |apa: &mut AuxParamsAttr| {
            apa.counter = apa.counter.wrapping_add(1);
            apa.has_expensive_expr_after_last_attr = false;
        };
        let mut ac = AttrChecker::new(self.cx, self.seen_types, self.type_cache);
        if ac.has_sig_drop_attr(self.cx.typeck_results().expr_ty(expr)) {
            if let hir::StmtKind::Local(local) = self.ap.curr_stmt.kind
                && let hir::PatKind::Binding(_, hir_id, ident, _) = local.pat.kind
                && !self.ap.apas.contains_key(&hir_id)
                && {
                    if let Some(local_hir_id) = path_to_local(expr) {
                        local_hir_id == hir_id
                    }
                    else {
                        true
                    }
                }
            {
                let mut apa = AuxParamsAttr {
                    first_bind_ident: ident,
                    first_block_hir_id: self.ap.curr_block_hir_id,
                    first_block_span: self.ap.curr_block_span,
                    first_method_span: {
                        let expr_or_init = expr_or_init(self.cx, expr);
                        if let hir::ExprKind::MethodCall(_, local_expr, _, span) = expr_or_init.kind {
                            local_expr.span.to(span)
                        }
                        else {
                            expr_or_init.span
                        }
                    },
                    first_stmt_span: self.ap.curr_stmt.span,
                    ..Default::default()
                };
                modify_apa_params(&mut apa);
                let _ = self.ap.apas.insert(hir_id, apa);
            } else {
                let Some(hir_id) = path_to_local(expr) else { return; };
                let Some(apa) = self.ap.apas.get_mut(&hir_id) else { return; };
                match self.ap.curr_stmt.kind {
                    hir::StmtKind::Local(local) => {
                        if let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind {
                            apa.last_bind_ident = ident;
                        }
                        if let Some(local_init) = local.init
                            && let hir::ExprKind::MethodCall(_, _, _, span) = local_init.kind
                        {
                            apa.last_method_span = span;
                        }
                    },
                    hir::StmtKind::Semi(expr) => {
                        if has_drop(expr, &apa.first_bind_ident) {
                            apa.has_expensive_expr_after_last_attr = false;
                            apa.last_stmt_span = DUMMY_SP;
                            return;
                        }
                        if let hir::ExprKind::MethodCall(_, _, _, span) = expr.kind {
                            apa.last_method_span = span;
                        }
                    },
                    _ => {},
                }
                apa.last_stmt_span = self.ap.curr_stmt.span;
                modify_apa_params(apa);
            }
        }
        walk_expr(self, expr);
    }
}

/// Auxiliary parameters used on each block check of an item
struct AuxParams<'others, 'stmt, 'tcx> {
    //// See [AuxParamsAttr].
    apas: &'others mut FxIndexMap<hir::HirId, AuxParamsAttr>,
    /// The current block identifier that is being visited.
    curr_block_hir_id: hir::HirId,
    /// The current block span that is being visited.
    curr_block_span: Span,
    /// The current statement that is being visited.
    curr_stmt: Cow<'stmt, hir::Stmt<'tcx>>,
}

impl<'others, 'stmt, 'tcx> AuxParams<'others, 'stmt, 'tcx> {
    fn new(apas: &'others mut FxIndexMap<hir::HirId, AuxParamsAttr>, curr_stmt: &'stmt hir::Stmt<'tcx>) -> Self {
        Self {
            apas,
            curr_block_hir_id: hir::HirId::INVALID,
            curr_block_span: DUMMY_SP,
            curr_stmt: Cow::Borrowed(curr_stmt),
        }
    }
}

/// Auxiliary parameters used on expression created with `#[has_significant_drop]`.
#[derive(Debug)]
struct AuxParamsAttr {
    /// The number of times `#[has_significant_drop]` was referenced.
    counter: usize,
    /// If an expensive expression follows the last use of anything marked with
    /// `#[has_significant_drop]`.
    has_expensive_expr_after_last_attr: bool,

    /// The identifier of the block that involves the first `#[has_significant_drop]`.
    first_block_hir_id: hir::HirId,
    /// The span of the block that involves the first `#[has_significant_drop]`.
    first_block_span: Span,
    /// The binding or variable that references the initial construction of the type marked with
    /// `#[has_significant_drop]`.
    first_bind_ident: Ident,
    /// Similar to `init_bind_ident` but encompasses the right-hand method call.
    first_method_span: Span,
    /// Similar to `init_bind_ident` but encompasses the whole contained statement.
    first_stmt_span: Span,

    /// The last visited binding or variable span within a block that had any referenced inner type
    /// marked with `#[has_significant_drop]`.
    last_bind_ident: Ident,
    /// Similar to `last_bind_span` but encompasses the right-hand method call.
    last_method_span: Span,
    /// Similar to `last_bind_span` but encompasses the whole contained statement.
    last_stmt_span: Span,
}

impl Default for AuxParamsAttr {
    fn default() -> Self {
        Self {
            counter: 0,
            has_expensive_expr_after_last_attr: false,
            first_block_hir_id: hir::HirId::INVALID,
            first_bind_ident: Ident::empty(),
            first_block_span: DUMMY_SP,
            first_method_span: DUMMY_SP,
            first_stmt_span: DUMMY_SP,
            last_bind_ident: Ident::empty(),
            last_method_span: DUMMY_SP,
            last_stmt_span: DUMMY_SP,
        }
    }
}

fn dummy_stmt_expr<'any>(expr: &'any hir::Expr<'any>) -> hir::Stmt<'any> {
    hir::Stmt {
        hir_id: hir::HirId::INVALID,
        kind: hir::StmtKind::Expr(expr),
        span: DUMMY_SP,
    }
}

fn has_drop(expr: &hir::Expr<'_>, first_bind_ident: &Ident) -> bool {
    if let hir::ExprKind::Call(fun, args) = expr.kind
        && let hir::ExprKind::Path(hir::QPath::Resolved(_, fun_path)) = &fun.kind
        && let [fun_ident, ..] = fun_path.segments
        && fun_ident.ident.name == rustc_span::sym::drop
        && let [first_arg, ..] = args
        && let hir::ExprKind::Path(hir::QPath::Resolved(_, arg_path)) = &first_arg.kind
        && let [first_arg_ps, .. ] = arg_path.segments
    {
        &first_arg_ps.ident == first_bind_ident
    }
    else {
        false
    }
}

fn is_expensive_expr(expr: &hir::Expr<'_>) -> bool {
    !matches!(expr.kind, hir::ExprKind::Path(_))
}
