use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::numeric_literal::NumericLiteral;
use clippy_utils::res::MaybeResPath as _;
use clippy_utils::source::{SpanRangeExt, snippet, snippet_with_applicability};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::visitors::{Visitable, for_each_expr_without_closures};
use clippy_utils::{get_parent_expr, is_hir_ty_cfg_dependant, is_ty_alias, sym};
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, FnRetTy, HirId, Lit, Node, Path, QPath, TyKind, UnOp};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, FloatTy, GenericArg, InferTy, Ty};
use rustc_span::Symbol;
use std::ops::ControlFlow;

use super::UNNECESSARY_CAST;

#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cast_expr: &Expr<'tcx>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
) -> bool {
    let mut app = Applicability::MachineApplicable;
    let cast_str = snippet_with_applicability(cx, cast_expr.span, "_", &mut app);

    if let ty::RawPtr(..) = cast_from.kind()
        // check both mutability and type are the same
        && cast_from.kind() == cast_to.kind()
        && let ExprKind::Cast(_, cast_to_hir) = expr.kind
        // Ignore casts to e.g. type aliases and infer types
        // - p as pointer_alias
        // - p as _
        && let TyKind::Ptr(to_pointee) = cast_to_hir.kind
    {
        match to_pointee.ty.kind {
            // Ignore casts to pointers that are aliases or cfg dependant, e.g.
            // - p as *const std::ffi::c_char (alias)
            // - p as *const std::os::raw::c_char (cfg dependant)
            TyKind::Path(qpath) if is_ty_alias(&qpath) || is_hir_ty_cfg_dependant(cx, to_pointee.ty) => {
                return false;
            },
            // Ignore `p as *const _`
            TyKind::Infer(()) => return false,
            _ => {},
        }

        // Preserve parentheses around `expr` in case of cascaded casts
        let surrounding =
            if matches!(cast_expr.kind, ExprKind::Cast(..)) && has_enclosing_paren(snippet(cx, expr.span, "")) {
                MaybeParenOrBlock::Paren
            } else {
                MaybeParenOrBlock::Nothing
            };

        emit_lint(
            cx,
            expr,
            format!(
                "casting raw pointers to the same type and constness is unnecessary (`{cast_from}` -> `{cast_to}`)"
            ),
            &cast_str,
            surrounding,
            app.max(Applicability::MaybeIncorrect),
        );
    }

    // skip cast of local that is a type alias
    if let ExprKind::Cast(inner, ..) = expr.kind
        && let ExprKind::Path(qpath) = inner.kind
        && let QPath::Resolved(None, Path { res, .. }) = qpath
        && let Res::Local(hir_id) = res
        && let parent = cx.tcx.parent_hir_node(*hir_id)
        && let Node::LetStmt(local) = parent
    {
        if let Some(ty) = local.ty
            && let TyKind::Path(qpath) = ty.kind
            && is_ty_alias(&qpath)
        {
            return false;
        }

        if let Some(expr) = local.init
            && let ExprKind::Cast(.., cast_to) = expr.kind
            && let TyKind::Path(qpath) = cast_to.kind
            && is_ty_alias(&qpath)
        {
            return false;
        }
    }

    // skip cast to non-primitive type
    if let ExprKind::Cast(_, cast_to) = expr.kind
        && let TyKind::Path(QPath::Resolved(_, path)) = &cast_to.kind
        && let Res::PrimTy(_) = path.res
    {
    } else {
        return false;
    }

    // skip cast of fn call that returns type alias
    if let ExprKind::Cast(inner, ..) = expr.kind
        && is_cast_from_ty_alias(cx, inner)
    {
        return false;
    }

    if let Some(lit) = get_numeric_literal(cast_expr) {
        let literal_str = &cast_str;

        if let LitKind::Int(n, _) = lit.node
            && let Some(src) = cast_expr.span.get_source_text(cx)
            && cast_to.is_floating_point()
            && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node)
            && let from_nbits = 128 - n.get().leading_zeros()
            && let to_nbits = fp_ty_mantissa_nbits(cast_to)
            && from_nbits != 0
            && to_nbits != 0
            && from_nbits <= to_nbits
            && num_lit.is_decimal()
        {
            lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
            return true;
        }

        match lit.node {
            LitKind::Int(_, LitIntType::Unsuffixed) if cast_to.is_integral() => {
                lint_unnecessary_cast(cx, expr, literal_str, cast_from, cast_to);
                return false;
            },
            LitKind::Float(_, LitFloatType::Unsuffixed) if cast_to.is_floating_point() => {
                lint_unnecessary_cast(cx, expr, literal_str, cast_from, cast_to);
                return false;
            },
            LitKind::Int(_, LitIntType::Signed(_) | LitIntType::Unsigned(_))
            | LitKind::Float(_, LitFloatType::Suffixed(_))
                if cast_from.kind() == cast_to.kind() =>
            {
                if let Some(src) = cast_expr.span.get_source_text(cx)
                    && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node)
                {
                    lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
                    return true;
                }
            },
            _ => {},
        }
    }

    if cast_from.kind() == cast_to.kind() && !expr.span.in_external_macro(cx.sess().source_map()) {
        fn is_borrow_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            matches!(expr.kind, ExprKind::AddrOf(..))
                || cx
                    .typeck_results()
                    .expr_adjustments(expr)
                    .first()
                    .is_some_and(|adj| matches!(adj.kind, Adjust::Borrow(_)))
        }

        fn is_in_allowed_macro(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            const ALLOWED_MACROS: &[Symbol] = &[
                sym::format_args_macro,
                sym::assert_eq_macro,
                sym::debug_assert_eq_macro,
                sym::assert_ne_macro,
                sym::debug_assert_ne_macro,
            ];
            matches!(expr.span.ctxt().outer_expn_data().macro_def_id, Some(def_id) if
                cx.tcx.get_diagnostic_name(def_id).is_some_and(|sym| ALLOWED_MACROS.contains(&sym)))
        }

        // Removing the cast here can change inference along the path to an outer
        // method receiver, so avoid linting in that case.
        if is_inference_sensitive_inner_expr(cx, cast_expr)
            && contains_unsuffixed_numeric_literal(cast_expr)
            && feeds_outer_method_receiver(cx, expr)
            && has_lint_blocking_context_on_receiver_path(cx, expr)
        {
            return false;
        }

        if let Some(id) = cast_expr.res_local_id()
            && !cx.tcx.hir_span(id).eq_ctxt(cast_expr.span)
        {
            // Binding context is different than the identifiers context.
            // Weird macro wizardry could be involved here.
            return false;
        }

        // Changing `&(x as i32)` to `&x` would change the meaning of the code because the previous creates
        // a reference to the temporary while the latter creates a reference to the original value.
        let surrounding = match cx.tcx.parent_hir_node(expr.hir_id) {
            Node::Expr(parent) if is_borrow_expr(cx, parent) && !is_in_allowed_macro(cx, parent) => {
                MaybeParenOrBlock::Block
            },
            Node::Expr(parent) if cx.precedence(cast_expr) < cx.precedence(parent) => MaybeParenOrBlock::Paren,
            _ => MaybeParenOrBlock::Nothing,
        };

        emit_lint(
            cx,
            expr,
            format!("casting to the same type is unnecessary (`{cast_from}` -> `{cast_to}`)"),
            &cast_str,
            surrounding,
            app,
        );
        return true;
    }

    false
}

fn lint_unnecessary_cast(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    raw_literal_str: &str,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
) {
    let literal_kind_name = if cast_from.is_integral() { "integer" } else { "float" };
    // first we remove all matches so `-(1)` become `-1`, and remove trailing dots, so `1.` become `1`
    let literal_str = raw_literal_str
        .replace(['(', ')'], "")
        .trim_end_matches('.')
        .to_string();
    // we know need to check if the parent is a method call, to add parenthesis accordingly (eg:
    // (-1).foo() instead of -1.foo())
    let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(..) = parent_expr.kind
        && literal_str.starts_with('-')
    {
        format!("({literal_str}_{cast_to})")
    } else {
        format!("{literal_str}_{cast_to}")
    };

    span_lint_and_sugg(
        cx,
        UNNECESSARY_CAST,
        expr.span,
        format!("casting {literal_kind_name} literal to `{cast_to}` is unnecessary"),
        "try",
        sugg,
        Applicability::MachineApplicable,
    );
}

fn get_numeric_literal<'e>(expr: &'e Expr<'e>) -> Option<Lit> {
    match expr.kind {
        ExprKind::Lit(lit) => Some(lit),
        ExprKind::Unary(UnOp::Neg, e) => {
            if let ExprKind::Lit(lit) = e.kind {
                Some(lit)
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Returns the mantissa bits wide of a fp type.
/// Will return 0 if the type is not a fp
fn fp_ty_mantissa_nbits(typ: Ty<'_>) -> u32 {
    match typ.kind() {
        ty::Float(FloatTy::F32) => 23,
        ty::Float(FloatTy::F64) | ty::Infer(InferTy::FloatVar(_)) => 52,
        _ => 0,
    }
}

/// Finds whether an `Expr` returns a type alias.
///
/// When in doubt, for example because it calls a non-local function that we don't have the
/// declaration for, assume if might be a type alias.
fn is_cast_from_ty_alias<'tcx>(cx: &LateContext<'tcx>, expr: impl Visitable<'tcx>) -> bool {
    for_each_expr_without_closures(expr, |expr| {
        // Calls are a `Path`, and usage of locals are a `Path`. So, this checks
        // - call() as i32
        // - local as i32
        if let ExprKind::Path(qpath) = expr.kind {
            let res = cx.qpath_res(&qpath, expr.hir_id);
            if let Res::Def(DefKind::Fn, def_id) = res {
                let Some(def_id) = def_id.as_local() else {
                    // External function, we can't know, better be safe
                    return ControlFlow::Break(());
                };
                if let Some(FnRetTy::Return(ty)) = cx.tcx.hir_get_fn_output(def_id)
                    && let TyKind::Path(qpath) = ty.kind
                    && is_ty_alias(&qpath)
                {
                    // Function call to a local function returning a type alias
                    return ControlFlow::Break(());
                }
            // Local usage
            } else if let Res::Local(hir_id) = res
                && let Node::LetStmt(l) = cx.tcx.parent_hir_node(hir_id)
            {
                if let Some(e) = l.init
                    && is_cast_from_ty_alias(cx, e)
                {
                    return ControlFlow::Break::<()>(());
                }

                if let Some(ty) = l.ty
                    && let TyKind::Path(qpath) = ty.kind
                    && is_ty_alias(&qpath)
                {
                    return ControlFlow::Break::<()>(());
                }
            }
        }

        ControlFlow::Continue(())
    })
    .is_some()
}

#[derive(Clone, Copy)]
enum MaybeParenOrBlock {
    Paren,
    Block,
    Nothing,
}

fn emit_lint(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    msg: String,
    sugg: &str,
    surrounding: MaybeParenOrBlock,
    applicability: Applicability,
) {
    span_lint_and_sugg(
        cx,
        UNNECESSARY_CAST,
        expr.span,
        msg,
        "try",
        match surrounding {
            MaybeParenOrBlock::Paren => format!("({sugg})"),
            MaybeParenOrBlock::Block => format!("{{ {sugg} }}"),
            MaybeParenOrBlock::Nothing => sugg.to_string(),
        },
        applicability,
    );
}

fn contains_unsuffixed_numeric_literal<'e>(expr: &'e Expr<'e>) -> bool {
    for_each_expr_without_closures(expr, |e| {
        if let Some(lit) = get_numeric_literal(e)
            && matches!(
                lit.node,
                LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed)
            )
        {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    })
    .is_some()
}

// Returns `true` for expressions whose resolved type or method depends on inference.
fn is_inference_sensitive_inner_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::MethodCall(..) | ExprKind::Binary(..) | ExprKind::Unary(..) | ExprKind::Index(..) => cx
            .typeck_results()
            .type_dependent_def_id(expr.hir_id)
            .and_then(|def_id| cx.tcx.opt_associated_item(def_id))
            .is_some_and(|assoc| assoc.trait_container(cx.tcx).is_some()),
        _ => false,
    }
}

// Returns `true` if the function's output type contains a type parameter
// originating from the selected input.
fn output_depends_on_input_param(cx: &LateContext<'_>, def_id: rustc_hir::def_id::DefId, input_index: usize) -> bool {
    let sig = cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();

    let Some(input_ty) = sig.inputs().get(input_index) else {
        return false;
    };

    let output_ty = sig.output();

    input_ty.walk().filter_map(GenericArg::as_type).any(|input_part| {
        match input_part.kind() {
            ty::Param(input_param) => output_ty
                .walk()
                .filter_map(GenericArg::as_type)
                .any(|output_part| {
                    matches!(output_part.kind(), ty::Param(output_param) if output_param.index == input_param.index)
                }),
            _ => false,
        }
    })
}

// Returns `true` if the generic arguments include at least one explicit type or const
// argument and none of the provided generic arguments are placeholders like `::<_>`.
fn has_explicit_type_or_const_args(args: Option<&rustc_hir::GenericArgs<'_>>) -> bool {
    let Some(args) = args else {
        return false;
    };

    let mut has_explicit = false;

    for arg in args.args {
        match arg {
            rustc_hir::GenericArg::Type(_) | rustc_hir::GenericArg::Const(_) => {
                has_explicit = true;
            },
            rustc_hir::GenericArg::Infer(_) => return false,
            rustc_hir::GenericArg::Lifetime(_) => {},
        }
    }

    has_explicit
}

// Controls whether the receiver path walk is looking for an outer method
// receiver or for a context where linting should stop.
#[derive(Copy, Clone)]
enum ReceiverPathMode {
    FindReceiver,
    FindLintBlockingContext,
}

enum ReceiverPathResult {
    Continue(HirId),
    Stop(bool),
}

fn stop_if_lint_blocking_else_continue(parent_hir_id: HirId, mode: ReceiverPathMode) -> ReceiverPathResult {
    if matches!(mode, ReceiverPathMode::FindLintBlockingContext) {
        ReceiverPathResult::Stop(true)
    } else {
        ReceiverPathResult::Continue(parent_hir_id)
    }
}

fn walk_receiver_path_method_call(
    cx: &LateContext<'_>,
    current_hir_id: HirId,
    parent: &Expr<'_>,
    segment: &rustc_hir::PathSegment<'_>,
    receiver: &Expr<'_>,
    args: &[Expr<'_>],
    mode: ReceiverPathMode,
) -> ReceiverPathResult {
    if receiver.hir_id == current_hir_id {
        return if matches!(mode, ReceiverPathMode::FindLintBlockingContext) {
            ReceiverPathResult::Continue(parent.hir_id)
        } else {
            ReceiverPathResult::Stop(true)
        };
    }

    let Some(arg_index) = args.iter().position(|arg| arg.hir_id == current_hir_id) else {
        return ReceiverPathResult::Stop(false);
    };

    let passthrough = !has_explicit_type_or_const_args(segment.args)
        && cx
            .typeck_results()
            .type_dependent_def_id(parent.hir_id)
            .is_some_and(|def_id| output_depends_on_input_param(cx, def_id, arg_index + 1));

    if matches!(mode, ReceiverPathMode::FindLintBlockingContext) {
        if passthrough
            || args.iter().any(|arg| {
                arg.hir_id != current_hir_id
                    && get_numeric_literal(arg).is_none()
                    && !cx.typeck_results().expr_ty(arg).is_primitive()
            })
        {
            ReceiverPathResult::Stop(true)
        } else {
            ReceiverPathResult::Continue(parent.hir_id)
        }
    } else if passthrough {
        ReceiverPathResult::Continue(parent.hir_id)
    } else {
        ReceiverPathResult::Stop(false)
    }
}

fn walk_receiver_path_call(
    cx: &LateContext<'_>,
    current_hir_id: HirId,
    parent: &Expr<'_>,
    callee: &Expr<'_>,
    args: &[Expr<'_>],
    mode: ReceiverPathMode,
) -> ReceiverPathResult {
    if callee.hir_id == current_hir_id {
        return ReceiverPathResult::Continue(parent.hir_id);
    }

    let Some(arg_index) = args.iter().position(|arg| arg.hir_id == current_hir_id) else {
        return ReceiverPathResult::Stop(false);
    };

    let passthrough = if let ExprKind::Path(qpath) = callee.kind
        && let Res::Def(DefKind::Fn, def_id) = cx.qpath_res(&qpath, callee.hir_id)
    {
        let has_explicit_args = match &qpath {
            QPath::Resolved(_, path) => path
                .segments
                .last()
                .is_some_and(|seg| has_explicit_type_or_const_args(seg.args)),
            QPath::TypeRelative(_, segment) => has_explicit_type_or_const_args(segment.args),
        };

        !has_explicit_args && output_depends_on_input_param(cx, def_id, arg_index)
    } else {
        false
    };

    if matches!(mode, ReceiverPathMode::FindLintBlockingContext) {
        ReceiverPathResult::Stop(passthrough)
    } else if passthrough {
        ReceiverPathResult::Continue(parent.hir_id)
    } else {
        ReceiverPathResult::Stop(false)
    }
}

// Walk one step up the receiver path for the current mode.
fn walk_receiver_path_step(cx: &LateContext<'_>, current_hir_id: HirId, mode: ReceiverPathMode) -> ReceiverPathResult {
    match cx.tcx.parent_hir_node(current_hir_id) {
        Node::Expr(parent) => match parent.kind {
            // Main case.
            // The current node may be the receiver.
            // Or it may flow through a passthrough method.
            ExprKind::MethodCall(segment, receiver, args, _) => {
                walk_receiver_path_method_call(cx, current_hir_id, parent, segment, receiver, args, mode)
            },
            // Regular calls only keep the path alive
            // if the output still depends on this input.
            ExprKind::Call(callee, args) => walk_receiver_path_call(cx, current_hir_id, parent, callee, args, mode),
            // A sibling that is not primitive blocks the lint.
            ExprKind::Binary(_, left, right) | ExprKind::Index(left, right, _)
                if left.hir_id == current_hir_id || right.hir_id == current_hir_id =>
            {
                if matches!(mode, ReceiverPathMode::FindLintBlockingContext) {
                    let sibling = if left.hir_id == current_hir_id { right } else { left };
                    if get_numeric_literal(sibling).is_none() && !cx.typeck_results().expr_ty(sibling).is_primitive() {
                        ReceiverPathResult::Stop(true)
                    } else {
                        ReceiverPathResult::Continue(parent.hir_id)
                    }
                } else {
                    ReceiverPathResult::Continue(parent.hir_id)
                }
            },
            // These expressions don't block the lint, so we continue walking up the path.
            ExprKind::Unary(_, inner)
            | ExprKind::Cast(inner, _)
            | ExprKind::AddrOf(_, _, inner)
            | ExprKind::Field(inner, _)
            | ExprKind::DropTemps(inner)
                if inner.hir_id == current_hir_id =>
            {
                ReceiverPathResult::Continue(parent.hir_id)
            },
            // A block can forward its tail expression, so we keep walking through it.
            ExprKind::Block(block, _)
                if block.hir_id == current_hir_id || block.expr.is_some_and(|tail| tail.hir_id == current_hir_id) =>
            {
                ReceiverPathResult::Continue(parent.hir_id)
            },
            // Depending on the mode, either keep walking or block the lint.
            ExprKind::Loop(block, ..) if block.hir_id == current_hir_id => {
                stop_if_lint_blocking_else_continue(parent.hir_id, mode)
            },
            // Tuples and arrays wrap the current expression, so we continue walking up the path.
            ExprKind::Tup(exprs) | ExprKind::Array(exprs) if exprs.iter().any(|e| e.hir_id == current_hir_id) => {
                ReceiverPathResult::Continue(parent.hir_id)
            },
            // The expression is stored in a field. We continue walking up the path to see how the struct is used.
            ExprKind::Struct(_, fields, _)
                if fields
                    .iter()
                    .any(|field| field.hir_id == current_hir_id || field.expr.hir_id == current_hir_id) =>
            {
                ReceiverPathResult::Continue(parent.hir_id)
            },
            // Depending on the mode, either keep walking or block the lint.
            ExprKind::If(cond, then_expr, else_expr)
                if cond.hir_id == current_hir_id
                    || then_expr.hir_id == current_hir_id
                    || else_expr.is_some_and(|else_expr| else_expr.hir_id == current_hir_id) =>
            {
                stop_if_lint_blocking_else_continue(parent.hir_id, mode)
            },
            // Depending on the mode, either keep walking or block the lint.
            ExprKind::Match(scrutinee, arms, _)
                if scrutinee.hir_id == current_hir_id
                    || arms
                        .iter()
                        .any(|arm| arm.hir_id == current_hir_id || arm.body.hir_id == current_hir_id) =>
            {
                stop_if_lint_blocking_else_continue(parent.hir_id, mode)
            },
            // Depending on the mode, either keep walking or block the lint.
            ExprKind::Break(_, Some(inner)) if inner.hir_id == current_hir_id => {
                stop_if_lint_blocking_else_continue(parent.hir_id, mode)
            },
            _ => ReceiverPathResult::Stop(false),
        },
        // These are structural HIR nodes. We just skip them and keep walking.
        Node::ExprField(_) | Node::Block(_) | Node::Arm(_) | Node::Stmt(_) => {
            ReceiverPathResult::Continue(cx.tcx.parent_hir_id(current_hir_id))
        },
        // Handle `let x = init; x` in the same block.
        // Depending on the mode, either keep walking init or block the lint.
        Node::LetStmt(local) => {
            if let Some(block_hir_id) = let_init_to_block_hir_id(cx, local, current_hir_id) {
                stop_if_lint_blocking_else_continue(block_hir_id, mode)
            } else {
                ReceiverPathResult::Stop(false)
            }
        },
        _ => ReceiverPathResult::Stop(false),
    }
}

// Returns `true` if `expr` eventually becomes the receiver of an outer method call.
fn feeds_outer_method_receiver(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let mut current_hir_id = expr.hir_id;

    loop {
        match walk_receiver_path_step(cx, current_hir_id, ReceiverPathMode::FindReceiver) {
            ReceiverPathResult::Continue(next) => current_hir_id = next,
            ReceiverPathResult::Stop(result) => return result,
        }
    }
}

// Returns `true` if the receiver path contains a context that should block the lint.
fn has_lint_blocking_context_on_receiver_path(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let mut current_hir_id = expr.hir_id;

    loop {
        match walk_receiver_path_step(cx, current_hir_id, ReceiverPathMode::FindLintBlockingContext) {
            ReceiverPathResult::Continue(next) => current_hir_id = next,
            ReceiverPathResult::Stop(result) => return result,
        }
    }
}

// If the initializer flows into the tail expression of the same block, returns that block HirId.
fn let_init_to_block_hir_id(
    cx: &LateContext<'_>,
    local: &rustc_hir::LetStmt<'_>,
    current_hir_id: HirId,
) -> Option<HirId> {
    let init = local.init?;
    if init.hir_id != current_hir_id {
        return None;
    }

    let stmt_hir_id = match cx.tcx.parent_hir_node(local.hir_id) {
        Node::Stmt(stmt) => stmt.hir_id,
        _ => return None,
    };

    let Node::Block(block) = cx.tcx.parent_hir_node(stmt_hir_id) else {
        return None;
    };

    let tail = block.expr?;
    let binding_hir_id = tail.res_local_id()?;

    match local.pat.kind {
        rustc_hir::PatKind::Binding(_, local_hir_id, ..) if local_hir_id == binding_hir_id => Some(block.hir_id),
        _ => None,
    }
}
