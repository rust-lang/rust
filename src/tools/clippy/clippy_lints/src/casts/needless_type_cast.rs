use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::visitors::{Descend, for_each_expr, for_each_expr_without_closures};
use core::ops::ControlFlow;
use rustc_ast::ast::{LitFloatType, LitIntType, LitKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{BlockCheckMode, Body, Expr, ExprKind, HirId, LetStmt, PatKind, StmtKind, UnsafeSource};
use rustc_lint::LateContext;
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::Span;

use super::NEEDLESS_TYPE_CAST;

struct BindingInfo<'a> {
    source_ty: Ty<'a>,
    ty_span: Span,
    init: Option<&'a Expr<'a>>,
}

struct UsageInfo<'a> {
    cast_to: Option<Ty<'a>>,
    in_generic_context: bool,
}

pub(super) fn check<'a>(cx: &LateContext<'a>, body: &Body<'a>) {
    let mut bindings: FxHashMap<HirId, BindingInfo<'a>> = FxHashMap::default();

    for_each_expr_without_closures(body.value, |expr| {
        match expr.kind {
            ExprKind::Block(block, _) => {
                for stmt in block.stmts {
                    if let StmtKind::Let(let_stmt) = stmt.kind {
                        collect_binding_from_local(cx, let_stmt, &mut bindings);
                    }
                }
            },
            ExprKind::Let(let_expr) => {
                collect_binding_from_let(cx, let_expr, &mut bindings);
            },
            _ => {},
        }
        ControlFlow::<()>::Continue(())
    });

    #[allow(rustc::potential_query_instability)]
    let mut binding_vec: Vec<_> = bindings.into_iter().collect();
    binding_vec.sort_by_key(|(_, info)| info.ty_span.lo());

    for (hir_id, binding_info) in binding_vec {
        check_binding_usages(cx, body, hir_id, &binding_info);
    }
}

fn collect_binding_from_let<'a>(
    cx: &LateContext<'a>,
    let_expr: &rustc_hir::LetExpr<'a>,
    bindings: &mut FxHashMap<HirId, BindingInfo<'a>>,
) {
    if let_expr.ty.is_none()
        || let_expr.span.from_expansion()
        || has_generic_return_type(cx, let_expr.init)
        || contains_unsafe(let_expr.init)
    {
        return;
    }

    if let PatKind::Binding(_, hir_id, _, _) = let_expr.pat.kind
        && let Some(ty_hir) = let_expr.ty
    {
        let ty = cx.typeck_results().pat_ty(let_expr.pat);
        if ty.is_numeric() {
            bindings.insert(
                hir_id,
                BindingInfo {
                    source_ty: ty,
                    ty_span: ty_hir.span,
                    init: Some(let_expr.init),
                },
            );
        }
    }
}

fn collect_binding_from_local<'a>(
    cx: &LateContext<'a>,
    let_stmt: &LetStmt<'a>,
    bindings: &mut FxHashMap<HirId, BindingInfo<'a>>,
) {
    if let_stmt.ty.is_none()
        || let_stmt.span.from_expansion()
        || let_stmt
            .init
            .is_some_and(|init| has_generic_return_type(cx, init) || contains_unsafe(init))
    {
        return;
    }

    if let PatKind::Binding(_, hir_id, _, _) = let_stmt.pat.kind
        && let Some(ty_hir) = let_stmt.ty
    {
        let ty = cx.typeck_results().pat_ty(let_stmt.pat);
        if ty.is_numeric() {
            bindings.insert(
                hir_id,
                BindingInfo {
                    source_ty: ty,
                    ty_span: ty_hir.span,
                    init: let_stmt.init,
                },
            );
        }
    }
}

fn contains_unsafe(expr: &Expr<'_>) -> bool {
    for_each_expr_without_closures(expr, |e| {
        if let ExprKind::Block(block, _) = e.kind
            && let BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) = block.rules
        {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })
    .is_some()
}

fn has_generic_return_type(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::Block(block, _) => {
            if let Some(tail_expr) = block.expr {
                return has_generic_return_type(cx, tail_expr);
            }
            false
        },
        ExprKind::If(_, then_block, else_expr) => {
            has_generic_return_type(cx, then_block) || else_expr.is_some_and(|e| has_generic_return_type(cx, e))
        },
        ExprKind::Match(_, arms, _) => arms.iter().any(|arm| has_generic_return_type(cx, arm.body)),
        ExprKind::Loop(block, label, ..) => for_each_expr_without_closures(*block, |e| {
            match e.kind {
                ExprKind::Loop(..) => {
                    // Unlabeled breaks inside nested loops target the inner loop, not ours
                    return ControlFlow::Continue(Descend::No);
                },
                ExprKind::Break(dest, Some(break_expr)) => {
                    let targets_this_loop =
                        dest.label.is_none() || dest.label.map(|l| l.ident) == label.map(|l| l.ident);
                    if targets_this_loop && has_generic_return_type(cx, break_expr) {
                        return ControlFlow::Break(());
                    }
                },
                _ => {},
            }
            ControlFlow::Continue(Descend::Yes)
        })
        .is_some(),
        ExprKind::MethodCall(..) => {
            if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
                let ret_ty = sig.output().skip_binder();
                return ret_ty.has_param();
            }
            false
        },
        ExprKind::Call(callee, _) => {
            if let ExprKind::Path(qpath) = &callee.kind {
                let res = cx.qpath_res(qpath, callee.hir_id);
                if let Res::Def(DefKind::Fn | DefKind::AssocFn, def_id) = res {
                    let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
                    let ret_ty = sig.output().skip_binder();
                    return ret_ty.has_param();
                }
            }
            false
        },
        _ => false,
    }
}

fn is_generic_res(cx: &LateContext<'_>, res: Res) -> bool {
    let has_type_params = |def_id| {
        cx.tcx
            .generics_of(def_id)
            .own_params
            .iter()
            .any(|p| p.kind.is_ty_or_const())
    };
    cx.tcx.res_generics_def_id(res).is_some_and(has_type_params)
}

fn is_cast_in_generic_context<'a>(cx: &LateContext<'a>, cast_expr: &Expr<'a>) -> bool {
    let mut current_id = cast_expr.hir_id;

    loop {
        let parent_id = cx.tcx.parent_hir_id(current_id);
        if parent_id == current_id {
            return false;
        }

        let parent = cx.tcx.hir_node(parent_id);

        match parent {
            rustc_hir::Node::Expr(parent_expr) => {
                match &parent_expr.kind {
                    ExprKind::Closure(_) => return false,
                    ExprKind::Call(callee, _) => {
                        if let ExprKind::Path(qpath) = &callee.kind {
                            let res = cx.qpath_res(qpath, callee.hir_id);
                            if is_generic_res(cx, res) {
                                return true;
                            }
                        }
                    },
                    ExprKind::MethodCall(..) => {
                        if let Some(def_id) = cx.typeck_results().type_dependent_def_id(parent_expr.hir_id)
                            && cx
                                .tcx
                                .generics_of(def_id)
                                .own_params
                                .iter()
                                .any(|p| p.kind.is_ty_or_const())
                        {
                            return true;
                        }
                    },
                    _ => {},
                }
                current_id = parent_id;
            },
            _ => return false,
        }
    }
}

fn can_coerce_to_target_type(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Lit(lit) => matches!(
            lit.node,
            LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed)
        ),
        ExprKind::Unary(rustc_hir::UnOp::Neg, inner) => can_coerce_to_target_type(inner),
        ExprKind::Binary(_, lhs, rhs) => can_coerce_to_target_type(lhs) && can_coerce_to_target_type(rhs),
        _ => false,
    }
}

fn check_binding_usages<'a>(cx: &LateContext<'a>, body: &Body<'a>, hir_id: HirId, binding_info: &BindingInfo<'a>) {
    let mut usages = Vec::new();

    for_each_expr(cx, body.value, |expr| {
        if let ExprKind::Path(ref qpath) = expr.kind
            && !expr.span.from_expansion()
            && let Res::Local(id) = cx.qpath_res(qpath, expr.hir_id)
            && id == hir_id
        {
            let parent_id = cx.tcx.parent_hir_id(expr.hir_id);
            let parent = cx.tcx.hir_node(parent_id);

            let usage = if let rustc_hir::Node::Expr(parent_expr) = parent
                && let ExprKind::Cast(..) = parent_expr.kind
                && !parent_expr.span.from_expansion()
            {
                UsageInfo {
                    cast_to: Some(cx.typeck_results().expr_ty(parent_expr)),
                    in_generic_context: is_cast_in_generic_context(cx, parent_expr),
                }
            } else {
                UsageInfo {
                    cast_to: None,
                    in_generic_context: false,
                }
            };
            usages.push(usage);
        }
        ControlFlow::<()>::Continue(())
    });

    let Some(first_target) = usages
        .first()
        .and_then(|u| u.cast_to)
        .filter(|&t| t != binding_info.source_ty)
        .filter(|&t| usages.iter().all(|u| u.cast_to == Some(t) && !u.in_generic_context))
    else {
        return;
    };

    // Don't lint if there's exactly one use and the initializer cannot be coerced to the
    // target type (i.e., would require an explicit cast). In such cases, the fix would add
    // a cast to the initializer rather than eliminating one - the cast isn't truly "needless."
    // See: https://github.com/rust-lang/rust-clippy/issues/16240
    if usages.len() == 1
        && binding_info
            .init
            .is_some_and(|init| !can_coerce_to_target_type(init) && !init.span.from_expansion())
    {
        return;
    }

    span_lint_and_then(
        cx,
        NEEDLESS_TYPE_CAST,
        binding_info.ty_span,
        format!(
            "this binding is defined as `{}` but is always cast to `{}`",
            binding_info.source_ty, first_target
        ),
        |diag| {
            if let Some(init) = binding_info
                .init
                .filter(|i| !can_coerce_to_target_type(i) && !i.span.from_expansion())
            {
                let sugg = Sugg::hir(cx, init, "..").as_ty(first_target);
                diag.multipart_suggestion(
                    format!("consider defining it as `{first_target}` and casting the initializer"),
                    vec![
                        (binding_info.ty_span, first_target.to_string()),
                        (init.span, sugg.to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            } else {
                diag.span_suggestion(
                    binding_info.ty_span,
                    "consider defining it as",
                    first_target.to_string(),
                    Applicability::MachineApplicable,
                );
            }
        },
    );
}
