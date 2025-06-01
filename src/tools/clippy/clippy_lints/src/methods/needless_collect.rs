use std::ops::ControlFlow;

use super::NEEDLESS_COLLECT;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{
    get_type_diagnostic_name, has_non_owning_mutable_access, make_normalized_projection, make_projection,
};
use clippy_utils::{
    CaptureKind, can_move_expr_to_closure, fn_def_id, get_enclosing_block, higher, is_trait_method, path_to_local,
    path_to_local_id, sym,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr, walk_stmt};
use rustc_hir::{
    BindingMode, Block, Expr, ExprKind, HirId, HirIdSet, LetStmt, Mutability, Node, Pat, PatKind, Stmt, StmtKind,
};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, AssocTag, ClauseKind, EarlyBinder, GenericArg, GenericArgKind, Ty};
use rustc_span::Span;
use rustc_span::symbol::Ident;

const NEEDLESS_COLLECT_MSG: &str = "avoid using `collect()` when not needed";

#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    name_span: Span,
    collect_expr: &'tcx Expr<'_>,
    iter_expr: &'tcx Expr<'tcx>,
    call_span: Span,
) {
    let iter_ty = cx.typeck_results().expr_ty(iter_expr);
    if has_non_owning_mutable_access(cx, iter_ty) {
        return; // don't lint if the iterator has side effects
    }

    match cx.tcx.parent_hir_node(collect_expr.hir_id) {
        Node::Expr(parent) => {
            check_collect_into_intoiterator(cx, parent, collect_expr, call_span, iter_expr);

            if let ExprKind::MethodCall(name, _, args @ ([] | [_]), _) = parent.kind {
                let mut app = Applicability::MachineApplicable;
                let name = name.ident.as_str();
                let collect_ty = cx.typeck_results().expr_ty(collect_expr);

                let sugg: String = match name {
                    "len" => {
                        if let Some(adt) = collect_ty.ty_adt_def()
                            && matches!(
                                cx.tcx.get_diagnostic_name(adt.did()),
                                Some(sym::Vec | sym::VecDeque | sym::LinkedList | sym::BinaryHeap)
                            )
                        {
                            "count()".into()
                        } else {
                            return;
                        }
                    },
                    "is_empty"
                        if is_is_empty_sig(cx, parent.hir_id)
                            && iterates_same_ty(cx, cx.typeck_results().expr_ty(iter_expr), collect_ty) =>
                    {
                        "next().is_none()".into()
                    },
                    "contains" => {
                        if is_contains_sig(cx, parent.hir_id, iter_expr)
                            && let Some(arg) = args.first()
                        {
                            let (span, prefix) = if let ExprKind::AddrOf(_, _, arg) = arg.kind {
                                (arg.span, "")
                            } else {
                                (arg.span, "*")
                            };
                            let snip = snippet_with_applicability(cx, span, "??", &mut app);
                            format!("any(|x| x == {prefix}{snip})")
                        } else {
                            return;
                        }
                    },
                    _ => return,
                };

                span_lint_and_sugg(
                    cx,
                    NEEDLESS_COLLECT,
                    call_span.with_hi(parent.span.hi()),
                    NEEDLESS_COLLECT_MSG,
                    "replace with",
                    sugg,
                    app,
                );
            }
        },
        Node::LetStmt(l) => {
            if let PatKind::Binding(BindingMode::NONE | BindingMode::MUT, id, _, None) = l.pat.kind
                && let ty = cx.typeck_results().expr_ty(collect_expr)
                && matches!(
                    get_type_diagnostic_name(cx, ty),
                    Some(sym::Vec | sym::VecDeque | sym::BinaryHeap | sym::LinkedList)
                )
                && let iter_ty = cx.typeck_results().expr_ty(iter_expr)
                && let Some(block) = get_enclosing_block(cx, l.hir_id)
                && let Some(iter_calls) = detect_iter_and_into_iters(block, id, cx, get_captured_ids(cx, iter_ty))
                && let [iter_call] = &*iter_calls
            {
                let mut used_count_visitor = UsedCountVisitor { cx, id, count: 0 };
                walk_block(&mut used_count_visitor, block);
                if used_count_visitor.count > 1 {
                    return;
                }

                if let IterFunctionKind::IntoIter(hir_id) = iter_call.func
                    && !check_iter_expr_used_only_as_iterator(cx, hir_id, block)
                {
                    return;
                }

                // Suggest replacing iter_call with iter_replacement, and removing stmt
                let mut span = MultiSpan::from_span(name_span);
                span.push_span_label(iter_call.span, "the iterator could be used here instead");
                span_lint_hir_and_then(
                    cx,
                    NEEDLESS_COLLECT,
                    collect_expr.hir_id,
                    span,
                    NEEDLESS_COLLECT_MSG,
                    |diag| {
                        let iter_replacement =
                            format!("{}{}", Sugg::hir(cx, iter_expr, ".."), iter_call.get_iter_method(cx));
                        diag.multipart_suggestion(
                            iter_call.get_suggestion_text(),
                            vec![(l.span, String::new()), (iter_call.span, iter_replacement)],
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        },
        _ => (),
    }
}

/// checks for collecting into a (generic) method or function argument
/// taking an `IntoIterator`
fn check_collect_into_intoiterator<'tcx>(
    cx: &LateContext<'tcx>,
    parent: &'tcx Expr<'tcx>,
    collect_expr: &'tcx Expr<'tcx>,
    call_span: Span,
    iter_expr: &'tcx Expr<'tcx>,
) {
    if let Some(id) = fn_def_id(cx, parent) {
        let args = match parent.kind {
            ExprKind::Call(_, args) | ExprKind::MethodCall(_, _, args, _) => args,
            _ => &[],
        };
        // find the argument index of the `collect_expr` in the
        // function / method call
        if let Some(arg_idx) = args.iter().position(|e| e.hir_id == collect_expr.hir_id).map(|i| {
            if matches!(parent.kind, ExprKind::MethodCall(_, _, _, _)) {
                i + 1
            } else {
                i
            }
        }) {
            // extract the input types of the function/method call
            // that contains `collect_expr`
            let inputs = cx
                .tcx
                .liberate_late_bound_regions(id, cx.tcx.fn_sig(id).instantiate_identity())
                .inputs();

            // map IntoIterator generic bounds to their signature
            // types and check whether the argument type is an
            // `IntoIterator`
            if cx
                .tcx
                .param_env(id)
                .caller_bounds()
                .into_iter()
                .filter_map(|p| {
                    if let ClauseKind::Trait(t) = p.kind().skip_binder()
                        && cx.tcx.is_diagnostic_item(sym::IntoIterator, t.trait_ref.def_id)
                    {
                        Some(t.self_ty())
                    } else {
                        None
                    }
                })
                .any(|ty| ty == inputs[arg_idx])
            {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_COLLECT,
                    call_span.with_lo(iter_expr.span.hi()),
                    NEEDLESS_COLLECT_MSG,
                    "remove this call",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

/// Checks if the given method call matches the expected signature of `([&[mut]] self) -> bool`
fn is_is_empty_sig(cx: &LateContext<'_>, call_id: HirId) -> bool {
    cx.typeck_results().type_dependent_def_id(call_id).is_some_and(|id| {
        let sig = cx.tcx.fn_sig(id).instantiate_identity().skip_binder();
        sig.inputs().len() == 1 && sig.output().is_bool()
    })
}

/// Checks if `<iter_ty as Iterator>::Item` is the same as `<collect_ty as IntoIter>::Item`
fn iterates_same_ty<'tcx>(cx: &LateContext<'tcx>, iter_ty: Ty<'tcx>, collect_ty: Ty<'tcx>) -> bool {
    if let Some(iter_trait) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(into_iter_trait) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
        && let Some(iter_item_ty) =
            make_normalized_projection(cx.tcx, cx.typing_env(), iter_trait, sym::Item, [iter_ty])
        && let Some(into_iter_item_proj) = make_projection(cx.tcx, into_iter_trait, sym::Item, [collect_ty])
        && let Ok(into_iter_item_ty) = cx.tcx.try_normalize_erasing_regions(
            cx.typing_env(),
            Ty::new_projection_from_args(cx.tcx, into_iter_item_proj.def_id, into_iter_item_proj.args),
        )
    {
        iter_item_ty == into_iter_item_ty
    } else {
        false
    }
}

/// Checks if the given method call matches the expected signature of
/// `([&[mut]] self, &<iter_ty as Iterator>::Item) -> bool`
fn is_contains_sig(cx: &LateContext<'_>, call_id: HirId, iter_expr: &Expr<'_>) -> bool {
    let typeck = cx.typeck_results();
    if let Some(id) = typeck.type_dependent_def_id(call_id)
        && let sig = cx.tcx.fn_sig(id).instantiate_identity()
        && sig.skip_binder().output().is_bool()
        && let [_, search_ty] = *sig.skip_binder().inputs()
        && let ty::Ref(_, search_ty, Mutability::Not) = *cx
            .tcx
            .instantiate_bound_regions_with_erased(sig.rebind(search_ty))
            .kind()
        && let Some(iter_trait) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(iter_item) = cx.tcx.associated_items(iter_trait).find_by_ident_and_kind(
            cx.tcx,
            Ident::with_dummy_span(sym::Item),
            AssocTag::Type,
            iter_trait,
        )
        && let args = cx.tcx.mk_args(&[GenericArg::from(typeck.expr_ty_adjusted(iter_expr))])
        && let proj_ty = Ty::new_projection_from_args(cx.tcx, iter_item.def_id, args)
        && let Ok(item_ty) = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), proj_ty)
    {
        item_ty == EarlyBinder::bind(search_ty).instantiate(cx.tcx, cx.typeck_results().node_args(call_id))
    } else {
        false
    }
}

struct IterFunction {
    func: IterFunctionKind,
    span: Span,
}
impl IterFunction {
    fn get_iter_method(&self, cx: &LateContext<'_>) -> String {
        match &self.func {
            IterFunctionKind::IntoIter(_) => String::new(),
            IterFunctionKind::Len => String::from(".count()"),
            IterFunctionKind::IsEmpty => String::from(".next().is_none()"),
            IterFunctionKind::Contains(span) => {
                let s = snippet(cx, *span, "..");
                if let Some(stripped) = s.strip_prefix('&') {
                    format!(".any(|x| x == {stripped})")
                } else {
                    format!(".any(|x| x == *{s})")
                }
            },
        }
    }
    fn get_suggestion_text(&self) -> &'static str {
        match &self.func {
            IterFunctionKind::IntoIter(_) => {
                "use the original Iterator instead of collecting it and then producing a new one"
            },
            IterFunctionKind::Len => {
                "take the original Iterator's count instead of collecting it and finding the length"
            },
            IterFunctionKind::IsEmpty => {
                "check if the original Iterator has anything instead of collecting it and seeing if it's empty"
            },
            IterFunctionKind::Contains(_) => {
                "check if the original Iterator contains an element instead of collecting then checking"
            },
        }
    }
}
enum IterFunctionKind {
    IntoIter(HirId),
    Len,
    IsEmpty,
    Contains(Span),
}

struct IterFunctionVisitor<'a, 'tcx> {
    illegal_mutable_capture_ids: HirIdSet,
    current_mutably_captured_ids: HirIdSet,
    cx: &'a LateContext<'tcx>,
    uses: Vec<Option<IterFunction>>,
    hir_id_uses_map: FxHashMap<HirId, usize>,
    current_statement_hir_id: Option<HirId>,
    seen_other: bool,
    target: HirId,
}
impl<'tcx> Visitor<'tcx> for IterFunctionVisitor<'_, 'tcx> {
    fn visit_block(&mut self, block: &'tcx Block<'tcx>) {
        for (expr, hir_id) in block.stmts.iter().filter_map(get_expr_and_hir_id_from_stmt) {
            if check_loop_kind(expr).is_some() {
                continue;
            }
            self.visit_block_expr(expr, hir_id);
        }
        if let Some(expr) = block.expr {
            if let Some(loop_kind) = check_loop_kind(expr) {
                if let LoopKind::Conditional(block_expr) = loop_kind {
                    self.visit_block_expr(block_expr, None);
                }
            } else {
                self.visit_block_expr(expr, None);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        // Check function calls on our collection
        if let ExprKind::MethodCall(method_name, recv, args, _) = &expr.kind {
            if args.is_empty()
                && method_name.ident.name == sym::collect
                && is_trait_method(self.cx, expr, sym::Iterator)
            {
                self.current_mutably_captured_ids = get_captured_ids(self.cx, self.cx.typeck_results().expr_ty(recv));
                self.visit_expr(recv);
                return;
            }

            if path_to_local_id(recv, self.target) {
                if self
                    .illegal_mutable_capture_ids
                    .intersection(&self.current_mutably_captured_ids)
                    .next()
                    .is_none()
                {
                    if let Some(hir_id) = self.current_statement_hir_id {
                        self.hir_id_uses_map.insert(hir_id, self.uses.len());
                    }
                    match method_name.ident.name {
                        sym::into_iter => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::IntoIter(expr.hir_id),
                            span: expr.span,
                        })),
                        sym::len => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::Len,
                            span: expr.span,
                        })),
                        sym::is_empty => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::IsEmpty,
                            span: expr.span,
                        })),
                        sym::contains => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::Contains(args[0].span),
                            span: expr.span,
                        })),
                        _ => {
                            self.seen_other = true;
                            if let Some(hir_id) = self.current_statement_hir_id {
                                self.hir_id_uses_map.remove(&hir_id);
                            }
                        },
                    }
                }
                return;
            }

            if let Some(hir_id) = path_to_local(recv)
                && let Some(index) = self.hir_id_uses_map.remove(&hir_id)
            {
                if self
                    .illegal_mutable_capture_ids
                    .intersection(&self.current_mutably_captured_ids)
                    .next()
                    .is_none()
                {
                    if let Some(hir_id) = self.current_statement_hir_id {
                        self.hir_id_uses_map.insert(hir_id, index);
                    }
                } else {
                    self.uses[index] = None;
                }
            }
        }
        // Check if the collection is used for anything else
        if path_to_local_id(expr, self.target) {
            self.seen_other = true;
        } else {
            walk_expr(self, expr);
        }
    }
}

enum LoopKind<'tcx> {
    Conditional(&'tcx Expr<'tcx>),
    Loop,
}

fn check_loop_kind<'tcx>(expr: &Expr<'tcx>) -> Option<LoopKind<'tcx>> {
    if let Some(higher::WhileLet { let_expr, .. }) = higher::WhileLet::hir(expr) {
        return Some(LoopKind::Conditional(let_expr));
    }
    if let Some(higher::While { condition, .. }) = higher::While::hir(expr) {
        return Some(LoopKind::Conditional(condition));
    }
    if let Some(higher::ForLoop { arg, .. }) = higher::ForLoop::hir(expr) {
        return Some(LoopKind::Conditional(arg));
    }
    if let ExprKind::Loop { .. } = expr.kind {
        return Some(LoopKind::Loop);
    }

    None
}

impl<'tcx> IterFunctionVisitor<'_, 'tcx> {
    fn visit_block_expr(&mut self, expr: &'tcx Expr<'tcx>, hir_id: Option<HirId>) {
        self.current_statement_hir_id = hir_id;
        self.current_mutably_captured_ids = get_captured_ids(self.cx, self.cx.typeck_results().expr_ty(expr));
        self.visit_expr(expr);
    }
}

fn get_expr_and_hir_id_from_stmt<'v>(stmt: &'v Stmt<'v>) -> Option<(&'v Expr<'v>, Option<HirId>)> {
    match stmt.kind {
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => Some((expr, None)),
        StmtKind::Item(..) => None,
        StmtKind::Let(LetStmt { init, pat, .. }) => {
            if let PatKind::Binding(_, hir_id, ..) = pat.kind {
                init.map(|init_expr| (init_expr, Some(hir_id)))
            } else {
                init.map(|init_expr| (init_expr, None))
            }
        },
    }
}

struct UsedCountVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    id: HirId,
    count: usize,
}

impl<'tcx> Visitor<'tcx> for UsedCountVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if path_to_local_id(expr, self.id) {
            self.count += 1;
        } else {
            walk_expr(self, expr);
        }
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

/// Detect the occurrences of calls to `iter` or `into_iter` for the
/// given identifier
fn detect_iter_and_into_iters<'tcx: 'a, 'a>(
    block: &'tcx Block<'tcx>,
    id: HirId,
    cx: &'a LateContext<'tcx>,
    captured_ids: HirIdSet,
) -> Option<Vec<IterFunction>> {
    let mut visitor = IterFunctionVisitor {
        illegal_mutable_capture_ids: captured_ids,
        current_mutably_captured_ids: HirIdSet::default(),
        cx,
        uses: Vec::new(),
        hir_id_uses_map: FxHashMap::default(),
        current_statement_hir_id: None,
        seen_other: false,
        target: id,
    };
    visitor.visit_block(block);
    if visitor.seen_other {
        None
    } else {
        Some(visitor.uses.into_iter().flatten().collect())
    }
}

fn get_captured_ids(cx: &LateContext<'_>, ty: Ty<'_>) -> HirIdSet {
    fn get_captured_ids_recursive(cx: &LateContext<'_>, ty: Ty<'_>, set: &mut HirIdSet) {
        match ty.kind() {
            ty::Adt(_, generics) => {
                for generic in *generics {
                    if let GenericArgKind::Type(ty) = generic.kind() {
                        get_captured_ids_recursive(cx, ty, set);
                    }
                }
            },
            ty::Closure(def_id, _) => {
                let closure_hir_node = cx.tcx.hir_get_if_local(*def_id).unwrap();
                if let Node::Expr(closure_expr) = closure_hir_node {
                    can_move_expr_to_closure(cx, closure_expr)
                        .unwrap()
                        .into_iter()
                        .for_each(|(hir_id, capture_kind)| {
                            if matches!(capture_kind, CaptureKind::Ref(Mutability::Mut)) {
                                set.insert(hir_id);
                            }
                        });
                }
            },
            _ => (),
        }
    }

    let mut set = HirIdSet::default();

    get_captured_ids_recursive(cx, ty, &mut set);

    set
}

struct IteratorMethodCheckVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    hir_id_of_expr: HirId,
    hir_id_of_let_binding: Option<HirId>,
}

impl<'tcx> Visitor<'tcx> for IteratorMethodCheckVisitor<'_, 'tcx> {
    type Result = ControlFlow<()>;
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> ControlFlow<()> {
        if let ExprKind::MethodCall(_method_name, recv, _args, _) = &expr.kind
            && (recv.hir_id == self.hir_id_of_expr
                || self
                    .hir_id_of_let_binding
                    .is_some_and(|hid| path_to_local_id(recv, hid)))
            && !is_trait_method(self.cx, expr, sym::Iterator)
        {
            return ControlFlow::Break(());
        } else if let ExprKind::Assign(place, value, _span) = &expr.kind
            && value.hir_id == self.hir_id_of_expr
            && let Some(id) = path_to_local(place)
        {
            // our iterator was directly assigned to a variable
            self.hir_id_of_let_binding = Some(id);
        }
        walk_expr(self, expr)
    }
    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'tcx>) -> ControlFlow<()> {
        if let StmtKind::Let(LetStmt {
            init: Some(expr),
            pat:
                Pat {
                    kind: PatKind::Binding(BindingMode::NONE | BindingMode::MUT, id, _, None),
                    ..
                },
            ..
        }) = &stmt.kind
            && expr.hir_id == self.hir_id_of_expr
        {
            // our iterator was directly assigned to a variable
            self.hir_id_of_let_binding = Some(*id);
        }
        walk_stmt(self, stmt)
    }
}

fn check_iter_expr_used_only_as_iterator<'tcx>(
    cx: &LateContext<'tcx>,
    hir_id_of_expr: HirId,
    block: &'tcx Block<'tcx>,
) -> bool {
    let mut visitor = IteratorMethodCheckVisitor {
        cx,
        hir_id_of_expr,
        hir_id_of_let_binding: None,
    };
    visitor.visit_block(block).is_continue()
}
