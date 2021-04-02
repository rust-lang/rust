use super::NEEDLESS_COLLECT;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{is_type_diagnostic_item, match_type};
use clippy_utils::{is_trait_method, path_to_local_id, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_block, walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Block, Expr, ExprKind, GenericArg, HirId, Local, Pat, PatKind, QPath, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{MultiSpan, Span};

const NEEDLESS_COLLECT_MSG: &str = "avoid using `collect()` when not needed";

pub(super) fn check<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    check_needless_collect_direct_usage(expr, cx);
    check_needless_collect_indirect_usage(expr, cx);
}
fn check_needless_collect_direct_usage<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    if_chain! {
        if let ExprKind::MethodCall(method, _, args, _) = expr.kind;
        if let ExprKind::MethodCall(chain_method, method0_span, _, _) = args[0].kind;
        if chain_method.ident.name == sym!(collect) && is_trait_method(cx, &args[0], sym::Iterator);
        if let Some(generic_args) = chain_method.args;
        if let Some(GenericArg::Type(ref ty)) = generic_args.args.get(0);
        let ty = cx.typeck_results().node_type(ty.hir_id);
        if is_type_diagnostic_item(cx, ty, sym::vec_type)
            || is_type_diagnostic_item(cx, ty, sym::vecdeque_type)
            || match_type(cx, ty, &paths::BTREEMAP)
            || is_type_diagnostic_item(cx, ty, sym::hashmap_type);
        if let Some(sugg) = match &*method.ident.name.as_str() {
            "len" => Some("count()".to_string()),
            "is_empty" => Some("next().is_none()".to_string()),
            "contains" => {
                let contains_arg = snippet(cx, args[1].span, "??");
                let (arg, pred) = contains_arg
                    .strip_prefix('&')
                    .map_or(("&x", &*contains_arg), |s| ("x", s));
                Some(format!("any(|{}| x == {})", arg, pred))
            }
            _ => None,
        };
        then {
            span_lint_and_sugg(
                cx,
                NEEDLESS_COLLECT,
                method0_span.with_hi(expr.span.hi()),
                NEEDLESS_COLLECT_MSG,
                "replace with",
                sugg,
                Applicability::MachineApplicable,
            );
        }
    }
}

fn check_needless_collect_indirect_usage<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) {
    if let ExprKind::Block(block, _) = expr.kind {
        for stmt in block.stmts {
            if_chain! {
                if let StmtKind::Local(
                    Local { pat: Pat { hir_id: pat_id, kind: PatKind::Binding(_, _, ident, .. ), .. },
                    init: Some(init_expr), .. }
                ) = stmt.kind;
                if let ExprKind::MethodCall(method_name, collect_span, &[ref iter_source], ..) = init_expr.kind;
                if method_name.ident.name == sym!(collect) && is_trait_method(cx, init_expr, sym::Iterator);
                if let Some(generic_args) = method_name.args;
                if let Some(GenericArg::Type(ref ty)) = generic_args.args.get(0);
                if let ty = cx.typeck_results().node_type(ty.hir_id);
                if is_type_diagnostic_item(cx, ty, sym::vec_type) ||
                    is_type_diagnostic_item(cx, ty, sym::vecdeque_type) ||
                    match_type(cx, ty, &paths::LINKED_LIST);
                if let Some(iter_calls) = detect_iter_and_into_iters(block, *ident);
                if let [iter_call] = &*iter_calls;
                then {
                    let mut used_count_visitor = UsedCountVisitor {
                        cx,
                        id: *pat_id,
                        count: 0,
                    };
                    walk_block(&mut used_count_visitor, block);
                    if used_count_visitor.count > 1 {
                        return;
                    }

                    // Suggest replacing iter_call with iter_replacement, and removing stmt
                    let mut span = MultiSpan::from_span(collect_span);
                    span.push_span_label(iter_call.span, "the iterator could be used here instead".into());
                    span_lint_and_then(
                        cx,
                        super::NEEDLESS_COLLECT,
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
                                Applicability::MachineApplicable,// MaybeIncorrect,
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

struct IterFunctionVisitor {
    uses: Vec<IterFunction>,
    seen_other: bool,
    target: Ident,
}
impl<'tcx> Visitor<'tcx> for IterFunctionVisitor {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        // Check function calls on our collection
        if_chain! {
            if let ExprKind::MethodCall(method_name, _, args, _) = &expr.kind;
            if let Some(Expr { kind: ExprKind::Path(QPath::Resolved(_, path)), .. }) = args.get(0);
            if let &[name] = &path.segments;
            if name.ident == self.target;
            then {
                let len = sym!(len);
                let is_empty = sym!(is_empty);
                let contains = sym!(contains);
                match method_name.ident.name {
                    sym::into_iter => self.uses.push(
                        IterFunction { func: IterFunctionKind::IntoIter, span: expr.span }
                    ),
                    name if name == len => self.uses.push(
                        IterFunction { func: IterFunctionKind::Len, span: expr.span }
                    ),
                    name if name == is_empty => self.uses.push(
                        IterFunction { func: IterFunctionKind::IsEmpty, span: expr.span }
                    ),
                    name if name == contains => self.uses.push(
                        IterFunction { func: IterFunctionKind::Contains(args[1].span), span: expr.span }
                    ),
                    _ => self.seen_other = true,
                }
                return
            }
        }
        // Check if the collection is used for anything else
        if_chain! {
            if let Expr { kind: ExprKind::Path(QPath::Resolved(_, path)), .. } = expr;
            if let &[name] = &path.segments;
            if name.ident == self.target;
            then {
                self.seen_other = true;
            } else {
                walk_expr(self, expr);
            }
        }
    }

    type Map = Map<'tcx>;
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

struct UsedCountVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    id: HirId,
    count: usize,
}

impl<'a, 'tcx> Visitor<'tcx> for UsedCountVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if path_to_local_id(expr, self.id) {
            self.count += 1;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

/// Detect the occurrences of calls to `iter` or `into_iter` for the
/// given identifier
fn detect_iter_and_into_iters<'tcx>(block: &'tcx Block<'tcx>, identifier: Ident) -> Option<Vec<IterFunction>> {
    let mut visitor = IterFunctionVisitor {
        uses: Vec::new(),
        target: identifier,
        seen_other: false,
    };
    visitor.visit_block(block);
    if visitor.seen_other { None } else { Some(visitor.uses) }
}
