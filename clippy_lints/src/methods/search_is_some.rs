use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_trait_method, strip_pat_refs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{self, HirId, HirIdMap, HirIdSet, PatKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::hir::place::ProjectionKind;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty;
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;
use rustc_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};

use super::SEARCH_IS_SOME;

/// lint searching an Iterator followed by `is_some()`
/// or calling `find()` on a string followed by `is_some()` or `is_none()`
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    expr: &'tcx hir::Expr<'_>,
    search_method: &str,
    is_some: bool,
    search_recv: &hir::Expr<'_>,
    search_arg: &'tcx hir::Expr<'_>,
    is_some_recv: &hir::Expr<'_>,
    method_span: Span,
) {
    let option_check_method = if is_some { "is_some" } else { "is_none" };
    // lint if caller of search is an Iterator
    if is_trait_method(cx, is_some_recv, sym::Iterator) {
        let msg = format!(
            "called `{}()` after searching an `Iterator` with `{}`",
            option_check_method, search_method
        );
        let search_snippet = snippet(cx, search_arg.span, "..");
        if search_snippet.lines().count() <= 1 {
            // suggest `any(|x| ..)` instead of `any(|&x| ..)` for `find(|&x| ..).is_some()`
            // suggest `any(|..| *..)` instead of `any(|..| **..)` for `find(|..| **..).is_some()`
            let any_search_snippet = if_chain! {
                if search_method == "find";
                if let hir::ExprKind::Closure(_, _, body_id, ..) = search_arg.kind;
                let closure_body = cx.tcx.hir().body(body_id);
                if let Some(closure_arg) = closure_body.params.get(0);

                then {
                    if let hir::PatKind::Ref(..) = closure_arg.pat.kind {
                        Some((search_snippet.replacen('&', "", 1), None))
                    } else if let PatKind::Binding(..) = strip_pat_refs(closure_arg.pat).kind {
                        let mut visitor = DerefDelegate {
                            cx,
                            set: HirIdSet::default(),
                            deref_suggs: HirIdMap::default(),
                            borrow_suggs: HirIdMap::default()
                        };

                        let fn_def_id = cx.tcx.hir().local_def_id(search_arg.hir_id);
                        cx.tcx.infer_ctxt().enter(|infcx| {
                            ExprUseVisitor::new(
                                &mut visitor, &infcx, fn_def_id, cx.param_env, cx.typeck_results()
                            ).consume_body(closure_body);
                        });

                        let replacements = if visitor.set.is_empty() {
                            None
                        } else {
                            let mut deref_suggs = Vec::new();
                            let mut borrow_suggs = Vec::new();
                            for node in visitor.set {
                                let span = cx.tcx.hir().span(node);
                                if let Some(sugg) = visitor.deref_suggs.get(&node) {
                                    deref_suggs.push((span, sugg.clone()));
                                }
                                if let Some(sugg) = visitor.borrow_suggs.get(&node) {
                                    borrow_suggs.push((span, sugg.clone()));
                                }
                            }
                            Some((deref_suggs, borrow_suggs))
                        };
                        Some((search_snippet.to_string(), replacements))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // add note if not multi-line
            let (closure_snippet, replacements) = any_search_snippet
                .as_ref()
                .map_or((&*search_snippet, None), |s| (&s.0, s.1.clone()));
            let (span, help, sugg) = if is_some {
                (
                    method_span.with_hi(expr.span.hi()),
                    "use `any()` instead",
                    format!("any({})", closure_snippet),
                )
            } else {
                let iter = snippet(cx, search_recv.span, "..");
                (
                    expr.span,
                    "use `!_.any()` instead",
                    format!("!{}.any({})", iter, closure_snippet),
                )
            };

            span_lint_and_then(cx, SEARCH_IS_SOME, span, &msg, |db| {
                if let Some((deref_suggs, borrow_suggs)) = replacements {
                    db.span_suggestion(span, help, sugg, Applicability::MaybeIncorrect);

                    if !deref_suggs.is_empty() {
                        db.multipart_suggestion("...and remove deref", deref_suggs, Applicability::MaybeIncorrect);
                    }
                    if !borrow_suggs.is_empty() {
                        db.multipart_suggestion("...and borrow variable", borrow_suggs, Applicability::MaybeIncorrect);
                    }
                } else {
                    db.span_suggestion(span, help, sugg, Applicability::MachineApplicable);
                }
            });
        } else {
            let hint = format!(
                "this is more succinctly expressed by calling `any()`{}",
                if option_check_method == "is_none" {
                    " with negation"
                } else {
                    ""
                }
            );
            span_lint_and_help(cx, SEARCH_IS_SOME, expr.span, &msg, None, &hint);
        }
    }
    // lint if `find()` is called by `String` or `&str`
    else if search_method == "find" {
        let is_string_or_str_slice = |e| {
            let self_ty = cx.typeck_results().expr_ty(e).peel_refs();
            if is_type_diagnostic_item(cx, self_ty, sym::String) {
                true
            } else {
                *self_ty.kind() == ty::Str
            }
        };
        if_chain! {
            if is_string_or_str_slice(search_recv);
            if is_string_or_str_slice(search_arg);
            then {
                let msg = format!("called `{}()` after calling `find()` on a string", option_check_method);
                match option_check_method {
                    "is_some" => {
                        let mut applicability = Applicability::MachineApplicable;
                        let find_arg = snippet_with_applicability(cx, search_arg.span, "..", &mut applicability);
                        span_lint_and_sugg(
                            cx,
                            SEARCH_IS_SOME,
                            method_span.with_hi(expr.span.hi()),
                            &msg,
                            "use `contains()` instead",
                            format!("contains({})", find_arg),
                            applicability,
                        );
                    },
                    "is_none" => {
                        let string = snippet(cx, search_recv.span, "..");
                        let mut applicability = Applicability::MachineApplicable;
                        let find_arg = snippet_with_applicability(cx, search_arg.span, "..", &mut applicability);
                        span_lint_and_sugg(
                            cx,
                            SEARCH_IS_SOME,
                            expr.span,
                            &msg,
                            "use `!_.contains()` instead",
                            format!("!{}.contains({})", string, find_arg),
                            applicability,
                        );
                    },
                    _ => (),
                }
            }
        }
    }
}

struct DerefDelegate<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    set: HirIdSet,
    deref_suggs: HirIdMap<String>,
    borrow_suggs: HirIdMap<String>,
}

impl<'tcx> Delegate<'tcx> for DerefDelegate<'_, 'tcx> {
    fn consume(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if let PlaceBase::Local(id) = cmt.place.base {
            let map = self.cx.tcx.hir();
            if cmt.place.projections.is_empty() {
                self.set.insert(cmt.hir_id);
            } else {
                let mut replacement_str = map.name(id).to_string();
                let last_deref = cmt
                    .place
                    .projections
                    .iter()
                    .rposition(|proj| proj.kind == ProjectionKind::Deref);

                if let Some(pos) = last_deref {
                    let mut projections = cmt.place.projections.clone();
                    projections.truncate(pos);

                    for item in projections {
                        if item.kind == ProjectionKind::Deref {
                            replacement_str = format!("*{}", replacement_str);
                        }
                    }

                    self.set.insert(cmt.hir_id);
                    self.deref_suggs.insert(cmt.hir_id, replacement_str);
                }
            }
        }
    }

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {
        if let PlaceBase::Local(id) = cmt.place.base {
            let map = self.cx.tcx.hir();
            if cmt.place.projections.is_empty() {
                let replacement_str = format!("&{}", map.name(id).to_string());
                self.set.insert(cmt.hir_id);
                self.borrow_suggs.insert(cmt.hir_id, replacement_str);
            } else {
                let mut replacement_str = map.name(id).to_string();
                let last_deref = cmt
                    .place
                    .projections
                    .iter()
                    .rposition(|proj| proj.kind == ProjectionKind::Deref);

                if let Some(pos) = last_deref {
                    let mut projections = cmt.place.projections.clone();
                    projections.truncate(pos);

                    for item in projections {
                        if item.kind == ProjectionKind::Deref {
                            replacement_str = format!("*{}", replacement_str);
                        }
                    }

                    self.set.insert(cmt.hir_id);
                    self.deref_suggs.insert(cmt.hir_id, replacement_str);
                }
            }
        }
    }

    fn mutate(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: rustc_typeck::expr_use_visitor::Place<'tcx>, _: FakeReadCause, _: HirId) {}
}
