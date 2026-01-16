use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::std_or_core;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{implements_trait, is_copy, peel_n_ty_refs};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::{Closure, Expr, ExprKind, Mutability, Param, Pat, PatKind, Path, PathSegment, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_span::symbol::Ident;
use rustc_span::{Span, sym};
use std::iter;
use std::ops::Not;

use super::UNNECESSARY_SORT_BY;

enum LintTrigger {
    Sort,
    SortByKey(SortByKeyDetection),
}

struct SortByKeyDetection {
    closure_arg: Span,
    closure_body: String,
    reverse: bool,
    applicability: Applicability,
}

/// Detect if the two expressions are mirrored (identical, except one
/// contains a and the other replaces it with b)
fn mirrored_exprs(
    a_expr: &Expr<'_>,
    b_expr: &Expr<'_>,
    binding_map: &BindingMap,
    binding_source: BindingSource,
) -> bool {
    match (a_expr.kind, b_expr.kind) {
        // Two arrays with mirrored contents
        (ExprKind::Array(left_exprs), ExprKind::Array(right_exprs)) => iter::zip(left_exprs, right_exprs)
            .all(|(left, right)| mirrored_exprs(left, right, binding_map, binding_source)),
        // The two exprs are function calls.
        // Check to see that the function itself and its arguments are mirrored
        (ExprKind::Call(left_expr, left_args), ExprKind::Call(right_expr, right_args)) => {
            mirrored_exprs(left_expr, right_expr, binding_map, binding_source)
                && iter::zip(left_args, right_args)
                    .all(|(left, right)| mirrored_exprs(left, right, binding_map, binding_source))
        },
        // The two exprs are method calls.
        // Check to see that the function is the same and the arguments and receivers are mirrored
        (
            ExprKind::MethodCall(left_segment, left_receiver, left_args, _),
            ExprKind::MethodCall(right_segment, right_receiver, right_args, _),
        ) => {
            left_segment.ident == right_segment.ident
                && iter::zip(left_args, right_args)
                    .all(|(left, right)| mirrored_exprs(left, right, binding_map, binding_source))
                && mirrored_exprs(left_receiver, right_receiver, binding_map, binding_source)
        },
        // Two tuples with mirrored contents
        (ExprKind::Tup(left_exprs), ExprKind::Tup(right_exprs)) => iter::zip(left_exprs, right_exprs)
            .all(|(left, right)| mirrored_exprs(left, right, binding_map, binding_source)),
        // Two binary ops, which are the same operation and which have mirrored arguments
        (ExprKind::Binary(left_op, left_left, left_right), ExprKind::Binary(right_op, right_left, right_right)) => {
            left_op.node == right_op.node
                && mirrored_exprs(left_left, right_left, binding_map, binding_source)
                && mirrored_exprs(left_right, right_right, binding_map, binding_source)
        },
        // Two unary ops, which are the same operation and which have the same argument
        (ExprKind::Unary(left_op, left_expr), ExprKind::Unary(right_op, right_expr)) => {
            left_op == right_op && mirrored_exprs(left_expr, right_expr, binding_map, binding_source)
        },
        // The two exprs are literals of some kind
        (ExprKind::Lit(left_lit), ExprKind::Lit(right_lit)) => left_lit.node == right_lit.node,
        (ExprKind::Cast(left, _), ExprKind::Cast(right, _)) => mirrored_exprs(left, right, binding_map, binding_source),
        (ExprKind::DropTemps(left_block), ExprKind::DropTemps(right_block)) => {
            mirrored_exprs(left_block, right_block, binding_map, binding_source)
        },
        (ExprKind::Field(left_expr, left_ident), ExprKind::Field(right_expr, right_ident)) => {
            left_ident.name == right_ident.name && mirrored_exprs(left_expr, right_expr, binding_map, binding_source)
        },
        // Two paths: either one is a and the other is b, or they're identical to each other
        (
            ExprKind::Path(QPath::Resolved(
                _,
                &Path {
                    segments: left_segments,
                    ..
                },
            )),
            ExprKind::Path(QPath::Resolved(
                _,
                &Path {
                    segments: right_segments,
                    ..
                },
            )),
        ) => {
            (iter::zip(left_segments, right_segments).all(|(left, right)| left.ident == right.ident)
                && left_segments.iter().all(|seg| {
                    !binding_map.contains_key(&BindingKey {
                        ident: seg.ident,
                        source: BindingSource::Left,
                    }) && !binding_map.contains_key(&BindingKey {
                        ident: seg.ident,
                        source: BindingSource::Right,
                    })
                }))
                || (left_segments.len() == 1
                    && right_segments.len() == 1
                    && binding_map
                        .get(&BindingKey {
                            ident: left_segments[0].ident,
                            source: binding_source,
                        })
                        .is_some_and(|value| value.mirrored.ident == right_segments[0].ident))
        },
        // Matching expressions, but one or both is borrowed
        (
            ExprKind::AddrOf(left_kind, Mutability::Not, left_expr),
            ExprKind::AddrOf(right_kind, Mutability::Not, right_expr),
        ) => left_kind == right_kind && mirrored_exprs(left_expr, right_expr, binding_map, binding_source),
        (_, ExprKind::AddrOf(_, Mutability::Not, right_expr)) => {
            mirrored_exprs(a_expr, right_expr, binding_map, binding_source)
        },
        (ExprKind::AddrOf(_, Mutability::Not, left_expr), _) => {
            mirrored_exprs(left_expr, b_expr, binding_map, binding_source)
        },
        _ => false,
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum BindingSource {
    Left,
    Right,
}

impl Not for BindingSource {
    type Output = BindingSource;

    fn not(self) -> Self::Output {
        match self {
            BindingSource::Left => BindingSource::Right,
            BindingSource::Right => BindingSource::Left,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct BindingKey {
    /// The identifier of the binding.
    ident: Ident,
    /// The source of the binding.
    source: BindingSource,
}

struct BindingValue {
    /// The mirrored binding.
    mirrored: BindingKey,
    /// The number of refs the binding is wrapped in.
    n_refs: usize,
}

/// A map from binding info to the number of refs the binding is wrapped in.
type BindingMap = FxHashMap<BindingKey, BindingValue>;
/// Extract the binding pairs, if the two patterns are mirrored. The pats are assumed to be used in
/// closure inputs and thus irrefutable.
fn mapping_of_mirrored_pats(a_pat: &Pat<'_>, b_pat: &Pat<'_>) -> Option<BindingMap> {
    fn mapping_of_mirrored_pats_inner(
        a_pat: &Pat<'_>,
        b_pat: &Pat<'_>,
        mapping: &mut BindingMap,
        n_refs: usize,
    ) -> bool {
        match (&a_pat.kind, &b_pat.kind) {
            (PatKind::Tuple(a_pats, a_dots), PatKind::Tuple(b_pats, b_dots)) => {
                a_dots == b_dots
                    && a_pats.len() == b_pats.len()
                    && iter::zip(a_pats.iter(), b_pats.iter())
                        .all(|(a, b)| mapping_of_mirrored_pats_inner(a, b, mapping, n_refs))
            },
            (PatKind::Binding(_, _, a_ident, _), PatKind::Binding(_, _, b_ident, _)) => {
                let a_key = BindingKey {
                    ident: *a_ident,
                    source: BindingSource::Left,
                };
                let b_key = BindingKey {
                    ident: *b_ident,
                    source: BindingSource::Right,
                };
                let a_value = BindingValue {
                    mirrored: b_key,
                    n_refs,
                };
                let b_value = BindingValue {
                    mirrored: a_key,
                    n_refs,
                };
                mapping.insert(a_key, a_value);
                mapping.insert(b_key, b_value);
                true
            },
            (PatKind::Wild, PatKind::Wild) => true,
            (PatKind::TupleStruct(_, a_pats, a_dots), PatKind::TupleStruct(_, b_pats, b_dots)) => {
                a_dots == b_dots
                    && a_pats.len() == b_pats.len()
                    && iter::zip(a_pats.iter(), b_pats.iter())
                        .all(|(a, b)| mapping_of_mirrored_pats_inner(a, b, mapping, n_refs))
            },
            (PatKind::Struct(_, a_fields, a_rest), PatKind::Struct(_, b_fields, b_rest)) => {
                a_rest == b_rest
                    && a_fields.len() == b_fields.len()
                    && iter::zip(a_fields.iter(), b_fields.iter()).all(|(a_field, b_field)| {
                        a_field.ident == b_field.ident
                            && mapping_of_mirrored_pats_inner(a_field.pat, b_field.pat, mapping, n_refs)
                    })
            },
            (PatKind::Ref(a_inner, _, _), PatKind::Ref(b_inner, _, _)) => {
                mapping_of_mirrored_pats_inner(a_inner, b_inner, mapping, n_refs + 1)
            },
            (PatKind::Slice(a_elems, None, a_rest), PatKind::Slice(b_elems, None, b_rest)) => {
                a_elems.len() == b_elems.len()
                    && iter::zip(a_elems.iter(), b_elems.iter())
                        .all(|(a, b)| mapping_of_mirrored_pats_inner(a, b, mapping, n_refs))
                    && a_rest.len() == b_rest.len()
                    && iter::zip(a_rest.iter(), b_rest.iter())
                        .all(|(a, b)| mapping_of_mirrored_pats_inner(a, b, mapping, n_refs))
            },
            _ => false,
        }
    }

    let mut mapping = FxHashMap::default();
    if mapping_of_mirrored_pats_inner(a_pat, b_pat, &mut mapping, 0) {
        return Some(mapping);
    }

    None
}

fn detect_lint(cx: &LateContext<'_>, expr: &Expr<'_>, arg: &Expr<'_>) -> Option<LintTrigger> {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && cx.tcx.type_of(impl_id).instantiate_identity().is_slice()
        && let ExprKind::Closure(&Closure { body, .. }) = arg.kind
        && let closure_body = cx.tcx.hir_body(body)
        && let &[Param { pat: l_pat, .. }, Param { pat: r_pat, .. }] = closure_body.params
        && let Some(binding_map) = mapping_of_mirrored_pats(l_pat, r_pat)
        && let ExprKind::MethodCall(method_path, left_expr, [right_expr], _) = closure_body.value.kind
        && method_path.ident.name == sym::cmp
        && let Some(ord_trait) = cx.tcx.get_diagnostic_item(sym::Ord)
        && cx.ty_based_def(closure_body.value).opt_parent(cx).opt_def_id() == Some(ord_trait)
    {
        let (closure_body, closure_arg, reverse) =
            if mirrored_exprs(left_expr, right_expr, &binding_map, BindingSource::Left) {
                (left_expr, l_pat.span, false)
            } else if mirrored_exprs(left_expr, right_expr, &binding_map, BindingSource::Right) {
                (left_expr, r_pat.span, true)
            } else {
                return None;
            };

        let mut applicability = if reverse {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        };

        if let ExprKind::Path(QPath::Resolved(
            _,
            Path {
                segments: [PathSegment { ident: left_name, .. }],
                ..
            },
        )) = left_expr.kind
        {
            if let PatKind::Binding(_, _, left_ident, _) = l_pat.kind
                && *left_name == left_ident
                && implements_trait(cx, cx.typeck_results().expr_ty(left_expr), ord_trait, &[])
            {
                return Some(LintTrigger::Sort);
            }

            let mut left_expr_ty = cx.typeck_results().expr_ty(left_expr);
            let left_ident_n_refs = binding_map
                .get(&BindingKey {
                    ident: *left_name,
                    source: BindingSource::Left,
                })
                .map_or(0, |value| value.n_refs);
            // Peel off the outer-most ref which is introduced by the closure, if it is not already peeled
            // by the pattern
            if left_ident_n_refs == 0 {
                (left_expr_ty, _) = peel_n_ty_refs(left_expr_ty, 1);
            }
            if !reverse && is_copy(cx, left_expr_ty) {
                let mut closure_body =
                    snippet_with_applicability(cx, closure_body.span, "_", &mut applicability).to_string();
                if left_ident_n_refs == 0 {
                    closure_body = format!("*{closure_body}");
                }
                return Some(LintTrigger::SortByKey(SortByKeyDetection {
                    closure_arg,
                    closure_body,
                    reverse,
                    applicability,
                }));
            }
        }

        let left_expr_ty = cx.typeck_results().expr_ty(left_expr);
        if !expr_borrows(left_expr_ty)
            // Don't lint if the closure is accessing non-Copy fields
            && (!expr_is_field_access(left_expr) || is_copy(cx, left_expr_ty))
        {
            let closure_body = Sugg::hir_with_applicability(cx, closure_body, "_", &mut applicability).to_string();
            return Some(LintTrigger::SortByKey(SortByKeyDetection {
                closure_arg,
                closure_body,
                reverse,
                applicability,
            }));
        }
    }

    None
}

fn expr_borrows(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Ref(..)) || ty.walk().any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(_)))
}

fn expr_is_field_access(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Field(_, _) => true,
        ExprKind::AddrOf(_, Mutability::Not, inner) => expr_is_field_access(inner),
        _ => false,
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
    is_unstable: bool,
) {
    match detect_lint(cx, expr, arg) {
        Some(LintTrigger::SortByKey(trigger)) => {
            let method = if is_unstable {
                "sort_unstable_by_key"
            } else {
                "sort_by_key"
            };
            let Some(std_or_core) = std_or_core(cx) else {
                // To make it this far the crate has to reference diagnostic items defined in core. Either this is
                // the `core` crate, there's an `extern crate core` somewhere, or another crate is defining the
                // diagnostic items. It's fine to not lint in all those cases even if we might be able to.
                return;
            };
            span_lint_and_then(
                cx,
                UNNECESSARY_SORT_BY,
                expr.span,
                format!("consider using `{method}`"),
                |diag| {
                    let mut app = trigger.applicability;
                    let recv = Sugg::hir_with_applicability(cx, recv, "(_)", &mut app);
                    let closure_body = if trigger.reverse {
                        format!("{std_or_core}::cmp::Reverse({})", trigger.closure_body)
                    } else {
                        trigger.closure_body
                    };
                    let closure_arg = snippet_with_applicability(cx, trigger.closure_arg, "_", &mut app);
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{recv}.{method}(|{closure_arg}| {closure_body})"),
                        app,
                    );
                },
            );
        },
        Some(LintTrigger::Sort) => {
            let method = if is_unstable { "sort_unstable" } else { "sort" };
            span_lint_and_then(
                cx,
                UNNECESSARY_SORT_BY,
                expr.span,
                format!("consider using `{method}`"),
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let recv = Sugg::hir_with_applicability(cx, recv, "(_)", &mut app);
                    diag.span_suggestion(expr.span, "try", format!("{recv}.{method}()"), app);
                },
            );
        },
        None => {},
    }
}
