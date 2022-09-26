use super::NEEDLESS_COLLECT;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::higher;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{can_move_expr_to_closure, is_trait_method, path_to_local, path_to_local_id, CaptureKind};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::intravisit::{walk_block, walk_expr, Visitor};
use rustc_hir::{Block, Expr, ExprKind, HirId, HirIdSet, Local, Mutability, Node, PatKind, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;
use rustc_span::Span;

const NEEDLESS_COLLECT_MSG: &str = "avoid using `collect()` when not needed";

pub(super) fn check<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    check_needless_collect_direct_usage(expr, cx);
    check_needless_collect_indirect_usage(expr, cx);
}
fn check_needless_collect_direct_usage<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    if_chain! {
        if let ExprKind::MethodCall(method, receiver, args, _) = expr.kind;
        if let ExprKind::MethodCall(chain_method, ..) = receiver.kind;
        if chain_method.ident.name == sym!(collect) && is_trait_method(cx, receiver, sym::Iterator);
        then {
            let ty = cx.typeck_results().expr_ty(receiver);
            let mut applicability = Applicability::MaybeIncorrect;
            let is_empty_sugg = "next().is_none()".to_string();
            let method_name = method.ident.name.as_str();
            let sugg = if is_type_diagnostic_item(cx, ty, sym::Vec) ||
                        is_type_diagnostic_item(cx, ty, sym::VecDeque) ||
                        is_type_diagnostic_item(cx, ty, sym::LinkedList) ||
                        is_type_diagnostic_item(cx, ty, sym::BinaryHeap) {
                match method_name {
                    "len" => "count()".to_string(),
                    "is_empty" => is_empty_sugg,
                    "contains" => {
                        let contains_arg = snippet_with_applicability(cx, args[0].span, "??", &mut applicability);
                        let (arg, pred) = contains_arg
                            .strip_prefix('&')
                            .map_or(("&x", &*contains_arg), |s| ("x", s));
                        format!("any(|{}| x == {})", arg, pred)
                    }
                    _ => return,
                }
            }
            else if is_type_diagnostic_item(cx, ty, sym::BTreeMap) ||
                is_type_diagnostic_item(cx, ty, sym::HashMap) {
                match method_name {
                    "is_empty" => is_empty_sugg,
                    _ => return,
                }
            }
            else {
                return;
            };
            span_lint_and_sugg(
                cx,
                NEEDLESS_COLLECT,
                chain_method.ident.span.with_hi(expr.span.hi()),
                NEEDLESS_COLLECT_MSG,
                "replace with",
                sugg,
                applicability,
            );
        }
    }
}

fn check_needless_collect_indirect_usage<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    if let ExprKind::Block(block, _) = expr.kind {
        for stmt in block.stmts {
            if_chain! {
                if let StmtKind::Local(local) = stmt.kind;
                if let PatKind::Binding(_, id, ..) = local.pat.kind;
                if let Some(init_expr) = local.init;
                if let ExprKind::MethodCall(method_name, iter_source, [], ..) = init_expr.kind;
                if method_name.ident.name == sym!(collect) && is_trait_method(cx, init_expr, sym::Iterator);
                let ty = cx.typeck_results().expr_ty(init_expr);
                if is_type_diagnostic_item(cx, ty, sym::Vec) ||
                    is_type_diagnostic_item(cx, ty, sym::VecDeque) ||
                    is_type_diagnostic_item(cx, ty, sym::BinaryHeap) ||
                    is_type_diagnostic_item(cx, ty, sym::LinkedList);
                let iter_ty = cx.typeck_results().expr_ty(iter_source);
                if let Some(iter_calls) = detect_iter_and_into_iters(block, id, cx, get_captured_ids(cx, iter_ty));
                if let [iter_call] = &*iter_calls;
                then {
                    let mut used_count_visitor = UsedCountVisitor {
                        cx,
                        id,
                        count: 0,
                    };
                    walk_block(&mut used_count_visitor, block);
                    if used_count_visitor.count > 1 {
                        return;
                    }

                    // Suggest replacing iter_call with iter_replacement, and removing stmt
                    let mut span = MultiSpan::from_span(method_name.ident.span);
                    span.push_span_label(iter_call.span, "the iterator could be used here instead");
                    span_lint_hir_and_then(
                        cx,
                        super::NEEDLESS_COLLECT,
                        init_expr.hir_id,
                        span,
                        NEEDLESS_COLLECT_MSG,
                        |diag| {
                            let iter_replacement = format!("{}{}", Sugg::hir(cx, iter_source, ".."), iter_call.get_iter_method(cx));
                            diag.multipart_suggestion(
                                iter_call.get_suggestion_text(),
                                vec![
                                    (stmt.span, String::new()),
                                    (iter_call.span, iter_replacement)
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        },
                    );
                }
            }
        }
    }
}

struct IterFunction {
    func: IterFunctionKind,
    span: Span,
}
impl IterFunction {
    fn get_iter_method(&self, cx: &LateContext<'_>) -> String {
        match &self.func {
            IterFunctionKind::IntoIter => String::new(),
            IterFunctionKind::Len => String::from(".count()"),
            IterFunctionKind::IsEmpty => String::from(".next().is_none()"),
            IterFunctionKind::Contains(span) => {
                let s = snippet(cx, *span, "..");
                if let Some(stripped) = s.strip_prefix('&') {
                    format!(".any(|x| x == {})", stripped)
                } else {
                    format!(".any(|x| x == *{})", s)
                }
            },
        }
    }
    fn get_suggestion_text(&self) -> &'static str {
        match &self.func {
            IterFunctionKind::IntoIter => {
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
    IntoIter,
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
        if let ExprKind::MethodCall(method_name, recv, [args @ ..], _) = &expr.kind {
            if method_name.ident.name == sym!(collect) && is_trait_method(self.cx, expr, sym::Iterator) {
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
                    match method_name.ident.name.as_str() {
                        "into_iter" => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::IntoIter,
                            span: expr.span,
                        })),
                        "len" => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::Len,
                            span: expr.span,
                        })),
                        "is_empty" => self.uses.push(Some(IterFunction {
                            func: IterFunctionKind::IsEmpty,
                            span: expr.span,
                        })),
                        "contains" => self.uses.push(Some(IterFunction {
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

            if let Some(hir_id) = path_to_local(recv) {
                if let Some(index) = self.hir_id_uses_map.remove(&hir_id) {
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
        StmtKind::Local(Local { init, pat, .. }) => {
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

impl<'a, 'tcx> Visitor<'tcx> for UsedCountVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if path_to_local_id(expr, self.id) {
            self.count += 1;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
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
        uses: Vec::new(),
        target: id,
        seen_other: false,
        cx,
        current_mutably_captured_ids: HirIdSet::default(),
        illegal_mutable_capture_ids: captured_ids,
        hir_id_uses_map: FxHashMap::default(),
        current_statement_hir_id: None,
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
                    if let GenericArgKind::Type(ty) = generic.unpack() {
                        get_captured_ids_recursive(cx, ty, set);
                    }
                }
            },
            ty::Closure(def_id, _) => {
                let closure_hir_node = cx.tcx.hir().get_if_local(*def_id).unwrap();
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
