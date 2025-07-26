use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{indent_of, snippet};
use clippy_utils::{expr_or_init, get_attr, path_to_local, peel_hir_expr_unary, sym};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{self as hir, HirId};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{GenericArgKind, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::Ident;
use rustc_span::{DUMMY_SP, Span};
use std::borrow::Cow;
use std::collections::hash_map::Entry;

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
    apas: FxIndexMap<HirId, AuxParamsAttr>,
    /// Auxiliary structure used to avoid having to verify the same type multiple times.
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
        StmtsChecker::new(&mut ap, cx, &mut self.type_cache).visit_body(body);
        for apa in ap.apas.values() {
            if apa.counter <= 1 || !apa.has_expensive_expr_after_last_attr {
                continue;
            }
            let first_bind_ident = apa.first_bind_ident.unwrap();
            span_lint_and_then(
                cx,
                SIGNIFICANT_DROP_TIGHTENING,
                first_bind_ident.span,
                "temporary with significant `Drop` can be early dropped",
                |diag| {
                    match apa.counter {
                        0 | 1 => {},
                        2 => {
                            let indent = " ".repeat(indent_of(cx, apa.last_stmt_span).unwrap_or(0));
                            let init_method = snippet(cx, apa.first_method_span, "..");
                            let usage_method = snippet(cx, apa.last_method_span, "..");
                            let stmt = if let Some(last_bind_ident) = apa.last_bind_ident {
                                format!(
                                    "\n{indent}let {} = {init_method}.{usage_method};",
                                    snippet(cx, last_bind_ident.span, ".."),
                                )
                            } else {
                                format!("\n{indent}{init_method}.{usage_method};")
                            };

                            diag.multipart_suggestion_verbose(
                                "merge the temporary construction with its single usage",
                                vec![(apa.first_stmt_span, stmt), (apa.last_stmt_span, String::new())],
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
                                    first_bind_ident
                                ),
                                Applicability::MaybeIncorrect,
                            );
                        },
                    }
                    diag.note("this might lead to unnecessary resource contention");
                    diag.span_label(
                        apa.first_block_span,
                        format!(
                            "temporary `{first_bind_ident}` is currently being dropped at the end of its contained scope"
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
    type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
}

impl<'cx, 'others, 'tcx> AttrChecker<'cx, 'others, 'tcx> {
    pub(crate) fn new(cx: &'cx LateContext<'tcx>, type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>) -> Self {
        Self { cx, type_cache }
    }

    fn has_sig_drop_attr(&mut self, ty: Ty<'tcx>, depth: usize) -> bool {
        if !self.cx.tcx.recursion_limit().value_within_limit(depth) {
            return false;
        }
        let ty = self
            .cx
            .tcx
            .try_normalize_erasing_regions(self.cx.typing_env(), ty)
            .unwrap_or(ty);
        match self.type_cache.entry(ty) {
            Entry::Occupied(e) => return *e.get(),
            Entry::Vacant(e) => {
                e.insert(false);
            },
        }
        let value = self.has_sig_drop_attr_uncached(ty, depth + 1);
        self.type_cache.insert(ty, value);
        value
    }

    fn has_sig_drop_attr_uncached(&mut self, ty: Ty<'tcx>, depth: usize) -> bool {
        if let Some(adt) = ty.ty_adt_def() {
            let mut iter = get_attr(
                self.cx.sess(),
                self.cx.tcx.get_attrs_unchecked(adt.did()),
                sym::has_significant_drop,
            );
            if iter.next().is_some() {
                return true;
            }
        }
        match ty.kind() {
            rustc_middle::ty::Adt(a, b) => {
                for f in a.all_fields() {
                    let ty = f.ty(self.cx.tcx, b);
                    if self.has_sig_drop_attr(ty, depth) {
                        return true;
                    }
                }
                for generic_arg in *b {
                    if let GenericArgKind::Type(ty) = generic_arg.kind()
                        && self.has_sig_drop_attr(ty, depth)
                    {
                        return true;
                    }
                }
                false
            },
            rustc_middle::ty::Array(ty, _)
            | rustc_middle::ty::RawPtr(ty, _)
            | rustc_middle::ty::Ref(_, ty, _)
            | rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr(*ty, depth),
            _ => false,
        }
    }
}

struct StmtsChecker<'ap, 'lc, 'others, 'stmt, 'tcx> {
    ap: &'ap mut AuxParams<'others, 'stmt, 'tcx>,
    cx: &'lc LateContext<'tcx>,
    type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
}

impl<'ap, 'lc, 'others, 'stmt, 'tcx> StmtsChecker<'ap, 'lc, 'others, 'stmt, 'tcx> {
    fn new(
        ap: &'ap mut AuxParams<'others, 'stmt, 'tcx>,
        cx: &'lc LateContext<'tcx>,
        type_cache: &'others mut FxHashMap<Ty<'tcx>, bool>,
    ) -> Self {
        Self { ap, cx, type_cache }
    }

    fn manage_has_expensive_expr_after_last_attr(&mut self) {
        let has_expensive_stmt = match self.ap.curr_stmt.kind {
            hir::StmtKind::Expr(expr) if is_inexpensive_expr(expr) => false,
            hir::StmtKind::Let(local)
                if let Some(expr) = local.init
                    && let hir::ExprKind::Path(_) = expr.kind =>
            {
                false
            },
            _ => true,
        };
        if has_expensive_stmt {
            for apa in self.ap.apas.values_mut() {
                let last_stmt_is_not_dummy = apa.last_stmt_span != DUMMY_SP;
                let last_stmt_is_not_curr = self.ap.curr_stmt.span != apa.last_stmt_span;
                let block_equals_curr = self.ap.curr_block_hir_id == apa.first_block_hir_id;
                let block_is_ancestor = self
                    .cx
                    .tcx
                    .hir_parent_iter(self.ap.curr_block_hir_id)
                    .any(|(id, _)| id == apa.first_block_hir_id);
                if last_stmt_is_not_dummy && last_stmt_is_not_curr && (block_equals_curr || block_is_ancestor) {
                    apa.has_expensive_expr_after_last_attr = true;
                }
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for StmtsChecker<'_, '_, '_, '_, 'tcx> {
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
        let mut ac = AttrChecker::new(self.cx, self.type_cache);
        if ac.has_sig_drop_attr(self.cx.typeck_results().expr_ty(expr), 0) {
            if let hir::StmtKind::Let(local) = self.ap.curr_stmt.kind
                && let hir::PatKind::Binding(_, hir_id, ident, _) = local.pat.kind
                && !self.ap.apas.contains_key(&hir_id)
                && {
                    if let Some(local_hir_id) = path_to_local(expr) {
                        local_hir_id == hir_id
                    } else {
                        true
                    }
                }
            {
                let mut apa = AuxParamsAttr {
                    first_block_hir_id: self.ap.curr_block_hir_id,
                    first_block_span: self.ap.curr_block_span,
                    first_bind_ident: Some(ident),
                    first_method_span: {
                        let expr_or_init = expr_or_init(self.cx, expr);
                        if let hir::ExprKind::MethodCall(_, local_expr, _, span) = expr_or_init.kind {
                            local_expr.span.to(span)
                        } else {
                            expr_or_init.span
                        }
                    },
                    first_stmt_span: self.ap.curr_stmt.span,
                    ..Default::default()
                };
                modify_apa_params(&mut apa);
                let _ = self.ap.apas.insert(hir_id, apa);
            } else {
                let Some(hir_id) = path_to_local(expr) else {
                    return;
                };
                let Some(apa) = self.ap.apas.get_mut(&hir_id) else {
                    return;
                };
                match self.ap.curr_stmt.kind {
                    hir::StmtKind::Let(local) => {
                        if let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind {
                            apa.last_bind_ident = Some(ident);
                        }
                        if let Some(local_init) = local.init
                            && let hir::ExprKind::MethodCall(_, _, _, span) = local_init.kind
                        {
                            apa.last_method_span = span;
                        }
                    },
                    hir::StmtKind::Semi(semi_expr) => {
                        if has_drop(semi_expr, apa.first_bind_ident, self.cx) {
                            apa.has_expensive_expr_after_last_attr = false;
                            apa.last_stmt_span = DUMMY_SP;
                            return;
                        }
                        if let hir::ExprKind::MethodCall(_, _, _, span) = semi_expr.kind {
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
    apas: &'others mut FxIndexMap<HirId, AuxParamsAttr>,
    /// The current block identifier that is being visited.
    curr_block_hir_id: HirId,
    /// The current block span that is being visited.
    curr_block_span: Span,
    /// The current statement that is being visited.
    curr_stmt: Cow<'stmt, hir::Stmt<'tcx>>,
}

impl<'others, 'stmt, 'tcx> AuxParams<'others, 'stmt, 'tcx> {
    fn new(apas: &'others mut FxIndexMap<HirId, AuxParamsAttr>, curr_stmt: &'stmt hir::Stmt<'tcx>) -> Self {
        Self {
            apas,
            curr_block_hir_id: HirId::INVALID,
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
    first_block_hir_id: HirId,
    /// The span of the block that involves the first `#[has_significant_drop]`.
    first_block_span: Span,
    /// The binding or variable that references the initial construction of the type marked with
    /// `#[has_significant_drop]`.
    first_bind_ident: Option<Ident>,
    /// Similar to `init_bind_ident` but encompasses the right-hand method call.
    first_method_span: Span,
    /// Similar to `init_bind_ident` but encompasses the whole contained statement.
    first_stmt_span: Span,

    /// The last visited binding or variable span within a block that had any referenced inner type
    /// marked with `#[has_significant_drop]`.
    last_bind_ident: Option<Ident>,
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
            first_block_hir_id: HirId::INVALID,
            first_block_span: DUMMY_SP,
            first_bind_ident: None,
            first_method_span: DUMMY_SP,
            first_stmt_span: DUMMY_SP,
            last_bind_ident: None,
            last_method_span: DUMMY_SP,
            last_stmt_span: DUMMY_SP,
        }
    }
}

fn dummy_stmt_expr<'any>(expr: &'any hir::Expr<'any>) -> hir::Stmt<'any> {
    hir::Stmt {
        hir_id: HirId::INVALID,
        kind: hir::StmtKind::Expr(expr),
        span: DUMMY_SP,
    }
}

fn has_drop(expr: &hir::Expr<'_>, first_bind_ident: Option<Ident>, lcx: &LateContext<'_>) -> bool {
    if let hir::ExprKind::Call(fun, [first_arg]) = expr.kind
        && let hir::ExprKind::Path(hir::QPath::Resolved(_, fun_path)) = &fun.kind
        && let Res::Def(DefKind::Fn, did) = fun_path.res
        && lcx.tcx.is_diagnostic_item(sym::mem_drop, did)
    {
        let has_ident = |local_expr: &hir::Expr<'_>| {
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, arg_path)) = &local_expr.kind
                && let [first_arg_ps, ..] = arg_path.segments
                && let Some(first_bind_ident) = first_bind_ident
                && first_arg_ps.ident == first_bind_ident
            {
                true
            } else {
                false
            }
        };
        if has_ident(first_arg) {
            return true;
        }
        if let hir::ExprKind::Tup(value) = &first_arg.kind
            && value.iter().any(has_ident)
        {
            return true;
        }
    }
    false
}

fn is_inexpensive_expr(expr: &hir::Expr<'_>) -> bool {
    let actual = peel_hir_expr_unary(expr).0;
    let is_path = matches!(actual.kind, hir::ExprKind::Path(_));
    let is_lit = matches!(actual.kind, hir::ExprKind::Lit(_));
    is_path || is_lit
}
