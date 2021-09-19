use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr_for_hir, is_trait_method, strip_pat_refs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{self, ExprKind, HirId, PatKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::hir::place::ProjectionKind;
use rustc_middle::mir::{FakeReadCause, Mutability};
use rustc_middle::ty;
use rustc_span::source_map::{BytePos, Span};
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
            let mut applicability = Applicability::MachineApplicable;
            let any_search_snippet = if_chain! {
                if search_method == "find";
                if let hir::ExprKind::Closure(_, _, body_id, ..) = search_arg.kind;
                let closure_body = cx.tcx.hir().body(body_id);
                if let Some(closure_arg) = closure_body.params.get(0);
                then {
                    if let hir::PatKind::Ref(..) = closure_arg.pat.kind {
                        Some(search_snippet.replacen('&', "", 1))
                    } else if let PatKind::Binding(_, binding_id, _, _) = strip_pat_refs(closure_arg.pat).kind {
                        // this binding is composed of at least two levels of references, so we need to remove one
                        let binding_type = cx.typeck_results().node_type(binding_id);
                        let innermost_is_ref = if let ty::Ref(_, inner,_) = binding_type.kind() {
                            matches!(inner.kind(), ty::Ref(_, innermost, _) if innermost.is_ref())
                        } else {
                            false
                        };

                        // `find()` provides a reference to the item, but `any` does not,
                        // so we should fix item usages for suggestion
                        if let Some(closure_sugg) = get_closure_suggestion(cx, search_arg, closure_body) {
                            applicability = closure_sugg.applicability;
                            if innermost_is_ref {
                                Some(closure_sugg.suggestion.replacen('&', "", 1))
                            } else {
                                Some(closure_sugg.suggestion)
                            }
                        } else if innermost_is_ref {
                            Some(search_snippet.replacen('&', "", 1))
                        } else {
                            Some(search_snippet.to_string())
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // add note if not multi-line
            if is_some {
                span_lint_and_sugg(
                    cx,
                    SEARCH_IS_SOME,
                    method_span.with_hi(expr.span.hi()),
                    &msg,
                    "use `any()` instead",
                    format!(
                        "any({})",
                        any_search_snippet.as_ref().map_or(&*search_snippet, String::as_str)
                    ),
                    applicability,
                );
            } else {
                let iter = snippet(cx, search_recv.span, "..");
                span_lint_and_sugg(
                    cx,
                    SEARCH_IS_SOME,
                    expr.span,
                    &msg,
                    "use `!_.any()` instead",
                    format!(
                        "!{}.any({})",
                        iter,
                        any_search_snippet.as_ref().map_or(&*search_snippet, String::as_str)
                    ),
                    applicability,
                );
            }
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

struct ClosureSugg {
    applicability: Applicability,
    suggestion: String,
}

// Build suggestion gradually by handling closure arg specific usages,
// such as explicit deref and borrowing cases.
// Returns `None` if no such use cases have been triggered in closure body
fn get_closure_suggestion<'tcx>(
    cx: &LateContext<'_>,
    search_arg: &'tcx hir::Expr<'_>,
    closure_body: &hir::Body<'_>,
) -> Option<ClosureSugg> {
    let mut visitor = DerefDelegate {
        cx,
        closure_span: search_arg.span,
        next_pos: search_arg.span.lo(),
        suggestion_start: String::new(),
        applicability: Applicability::MachineApplicable,
    };

    let fn_def_id = cx.tcx.hir().local_def_id(search_arg.hir_id);
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(&mut visitor, &infcx, fn_def_id, cx.param_env, cx.typeck_results())
            .consume_body(closure_body);
    });

    if visitor.suggestion_start.is_empty() {
        None
    } else {
        Some(ClosureSugg {
            applicability: visitor.applicability,
            suggestion: visitor.finish(),
        })
    }
}

struct DerefDelegate<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    closure_span: Span,
    next_pos: BytePos,
    suggestion_start: String,
    applicability: Applicability,
}

impl DerefDelegate<'_, 'tcx> {
    pub fn finish(&mut self) -> String {
        let end_span = Span::new(self.next_pos, self.closure_span.hi(), self.closure_span.ctxt());
        let end_snip = snippet_with_applicability(self.cx, end_span, "..", &mut self.applicability);
        format!("{}{}", self.suggestion_start, end_snip)
    }
}

impl<'tcx> Delegate<'tcx> for DerefDelegate<'_, 'tcx> {
    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {
        if let PlaceBase::Local(id) = cmt.place.base {
            let map = self.cx.tcx.hir();
            let ident_str = map.name(id).to_string();
            let span = map.span(cmt.hir_id);
            let start_span = Span::new(self.next_pos, span.lo(), span.ctxt());
            let mut start_snip = snippet_with_applicability(self.cx, start_span, "..", &mut self.applicability);

            if cmt.place.projections.is_empty() {
                // handle item without any projection, that needs an explicit borrowing
                // i.e.: suggest `&x` instead of `x`
                self.suggestion_start.push_str(&format!("{}&{}", start_snip, ident_str));
            } else {
                // cases where a parent call is using the item
                // i.e.: suggest `.contains(&x)` for `.find(|x| [1, 2, 3].contains(x)).is_none()`
                if let Some(parent_expr) = get_parent_expr_for_hir(self.cx, cmt.hir_id) {
                    if let ExprKind::Call(_, call_args) | ExprKind::MethodCall(_, _, call_args, _) = parent_expr.kind {
                        let expr = self.cx.tcx.hir().expect_expr(cmt.hir_id);
                        let arg_ty_kind = self.cx.typeck_results().expr_ty(expr).kind();

                        if matches!(arg_ty_kind, ty::Ref(_, _, Mutability::Not)) {
                            let start_span = Span::new(self.next_pos, span.lo(), span.ctxt());
                            let start_snip =
                                snippet_with_applicability(self.cx, start_span, "..", &mut self.applicability);
                            // do not suggest ampersand if the ident is the method caller
                            let ident_sugg = if !call_args.is_empty() && call_args[0].hir_id == cmt.hir_id {
                                format!("{}{}", start_snip, ident_str)
                            } else {
                                format!("{}&{}", start_snip, ident_str)
                            };
                            self.suggestion_start.push_str(&ident_sugg);
                            self.next_pos = span.hi();
                            return;
                        } else {
                            self.applicability = Applicability::Unspecified;
                        }
                    }
                }

                // handle item projections by removing one explicit deref
                // i.e.: suggest `*x` instead of `**x`
                let mut replacement_str = ident_str;

                // handle index projection first
                let index_handled = cmt.place.projections.iter().any(|proj| match proj.kind {
                    // Index projection like `|x| foo[x]`
                    // the index is dropped so we can't get it to build the suggestion,
                    // so the span is set-up again to get more code, using `span.hi()` (i.e.: `foo[x]`)
                    // instead of `span.lo()` (i.e.: `foo`)
                    ProjectionKind::Index => {
                        let start_span = Span::new(self.next_pos, span.hi(), span.ctxt());
                        start_snip = snippet_with_applicability(self.cx, start_span, "..", &mut self.applicability);
                        replacement_str.clear();
                        true
                    },
                    _ => false,
                });

                // looking for projections other that need to be handled differently
                let other_projections_handled = cmt.place.projections.iter().enumerate().any(|(i, proj)| {
                    match proj.kind {
                        // Field projection like `|v| v.foo`
                        ProjectionKind::Field(idx, variant) => match cmt.place.ty_before_projection(i).kind() {
                            ty::Adt(def, ..) => {
                                replacement_str = format!(
                                    "{}.{}",
                                    replacement_str,
                                    def.variants[variant].fields[idx as usize].ident.name.as_str()
                                );
                                true
                            },
                            ty::Tuple(_) => {
                                replacement_str = format!("{}.{}", replacement_str, idx);
                                true
                            },
                            _ => false,
                        },
                        // handled previously
                        ProjectionKind::Index |
                            // note: unable to trigger `Subslice` kind in tests
                            ProjectionKind::Subslice => false,
                        ProjectionKind::Deref => {
                            // explicit deref for arrays should be avoided in the suggestion
                            // i.e.: `|sub| *sub[1..4].len() == 3` is not expected
                            match cmt.place.ty_before_projection(i).kind() {
                                // dereferencing an array (i.e.: `|sub| sub[1..4].len() == 3`)
                                ty::Ref(_, inner, _) => {
                                    matches!(inner.kind(), ty::Ref(_, innermost, _) if innermost.is_array())
                                },
                                _ => false,
                            }
                        },
                    }
                });

                // handle `ProjectionKind::Deref` if no special case detected
                if !index_handled && !other_projections_handled {
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
                    }
                }

                self.suggestion_start
                    .push_str(&format!("{}{}", start_snip, replacement_str));
            }
            self.next_pos = span.hi();
        }
    }

    fn mutate(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: rustc_typeck::expr_use_visitor::Place<'tcx>, _: FakeReadCause, _: HirId) {}
}
