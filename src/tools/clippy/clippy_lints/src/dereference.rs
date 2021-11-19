use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::peel_mid_ty_refs;
use clippy_utils::{get_parent_node, in_macro, is_lint_allowed};
use rustc_ast::util::parser::PREC_PREFIX;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, HirId, MatchSource, Mutability, Node, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty, TyCtxt, TyS, TypeckResults};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit `deref()` or `deref_mut()` method calls.
    ///
    /// ### Why is this bad?
    /// Dereferencing by `&*x` or `&mut *x` is clearer and more concise,
    /// when not part of a method chain.
    ///
    /// ### Example
    /// ```rust
    /// use std::ops::Deref;
    /// let a: &mut String = &mut String::from("foo");
    /// let b: &str = a.deref();
    /// ```
    /// Could be written as:
    /// ```rust
    /// let a: &mut String = &mut String::from("foo");
    /// let b = &*a;
    /// ```
    ///
    /// This lint excludes
    /// ```rust,ignore
    /// let _ = d.unwrap().deref();
    /// ```
    pub EXPLICIT_DEREF_METHODS,
    pedantic,
    "Explicit use of deref or deref_mut method while not in a method chain."
}

impl_lint_pass!(Dereferencing => [
    EXPLICIT_DEREF_METHODS,
]);

#[derive(Default)]
pub struct Dereferencing {
    state: Option<(State, StateData)>,

    // While parsing a `deref` method call in ufcs form, the path to the function is itself an
    // expression. This is to store the id of that expression so it can be skipped when
    // `check_expr` is called for it.
    skip_expr: Option<HirId>,
}

struct StateData {
    /// Span of the top level expression
    span: Span,
    /// The required mutability
    target_mut: Mutability,
}

enum State {
    // Any number of deref method calls.
    DerefMethod {
        // The number of calls in a sequence which changed the referenced type
        ty_changed_count: usize,
        is_final_ufcs: bool,
    },
}

// A reference operation considered by this lint pass
enum RefOp {
    Method(Mutability),
    Deref,
    AddrOf,
}

impl<'tcx> LateLintPass<'tcx> for Dereferencing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Skip path expressions from deref calls. e.g. `Deref::deref(e)`
        if Some(expr.hir_id) == self.skip_expr.take() {
            return;
        }

        // Stop processing sub expressions when a macro call is seen
        if in_macro(expr.span) {
            if let Some((state, data)) = self.state.take() {
                report(cx, expr, state, data);
            }
            return;
        }

        let typeck = cx.typeck_results();
        let (kind, sub_expr) = if let Some(x) = try_parse_ref_op(cx.tcx, typeck, expr) {
            x
        } else {
            // The whole chain of reference operations has been seen
            if let Some((state, data)) = self.state.take() {
                report(cx, expr, state, data);
            }
            return;
        };

        match (self.state.take(), kind) {
            (None, kind) => {
                let parent = get_parent_node(cx.tcx, expr.hir_id);
                let expr_ty = typeck.expr_ty(expr);

                match kind {
                    RefOp::Method(target_mut)
                        if !is_lint_allowed(cx, EXPLICIT_DEREF_METHODS, expr.hir_id)
                            && is_linted_explicit_deref_position(parent, expr.hir_id, expr.span) =>
                    {
                        self.state = Some((
                            State::DerefMethod {
                                ty_changed_count: if deref_method_same_type(expr_ty, typeck.expr_ty(sub_expr)) {
                                    0
                                } else {
                                    1
                                },
                                is_final_ufcs: matches!(expr.kind, ExprKind::Call(..)),
                            },
                            StateData {
                                span: expr.span,
                                target_mut,
                            },
                        ));
                    },
                    _ => (),
                }
            },
            (Some((State::DerefMethod { ty_changed_count, .. }, data)), RefOp::Method(_)) => {
                self.state = Some((
                    State::DerefMethod {
                        ty_changed_count: if deref_method_same_type(typeck.expr_ty(expr), typeck.expr_ty(sub_expr)) {
                            ty_changed_count
                        } else {
                            ty_changed_count + 1
                        },
                        is_final_ufcs: matches!(expr.kind, ExprKind::Call(..)),
                    },
                    data,
                ));
            },

            (Some((state, data)), _) => report(cx, expr, state, data),
        }
    }
}

fn try_parse_ref_op(
    tcx: TyCtxt<'tcx>,
    typeck: &'tcx TypeckResults<'_>,
    expr: &'tcx Expr<'_>,
) -> Option<(RefOp, &'tcx Expr<'tcx>)> {
    let (def_id, arg) = match expr.kind {
        ExprKind::MethodCall(_, _, [arg], _) => (typeck.type_dependent_def_id(expr.hir_id)?, arg),
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(path),
                hir_id,
                ..
            },
            [arg],
        ) => (typeck.qpath_res(path, *hir_id).opt_def_id()?, arg),
        ExprKind::Unary(UnOp::Deref, sub_expr) if !typeck.expr_ty(sub_expr).is_unsafe_ptr() => {
            return Some((RefOp::Deref, sub_expr));
        },
        ExprKind::AddrOf(BorrowKind::Ref, _, sub_expr) => return Some((RefOp::AddrOf, sub_expr)),
        _ => return None,
    };
    if tcx.is_diagnostic_item(sym::deref_method, def_id) {
        Some((RefOp::Method(Mutability::Not), arg))
    } else if tcx.trait_of_item(def_id)? == tcx.lang_items().deref_mut_trait()? {
        Some((RefOp::Method(Mutability::Mut), arg))
    } else {
        None
    }
}

// Checks whether the type for a deref call actually changed the type, not just the mutability of
// the reference.
fn deref_method_same_type(result_ty: Ty<'tcx>, arg_ty: Ty<'tcx>) -> bool {
    match (result_ty.kind(), arg_ty.kind()) {
        (ty::Ref(_, result_ty, _), ty::Ref(_, arg_ty, _)) => TyS::same_type(result_ty, arg_ty),

        // The result type for a deref method is always a reference
        // Not matching the previous pattern means the argument type is not a reference
        // This means that the type did change
        _ => false,
    }
}

// Checks whether the parent node is a suitable context for switching from a deref method to the
// deref operator.
fn is_linted_explicit_deref_position(parent: Option<Node<'_>>, child_id: HirId, child_span: Span) -> bool {
    let parent = match parent {
        Some(Node::Expr(e)) if e.span.ctxt() == child_span.ctxt() => e,
        _ => return true,
    };
    match parent.kind {
        // Leave deref calls in the middle of a method chain.
        // e.g. x.deref().foo()
        ExprKind::MethodCall(_, _, [self_arg, ..], _) if self_arg.hir_id == child_id => false,

        // Leave deref calls resulting in a called function
        // e.g. (x.deref())()
        ExprKind::Call(func_expr, _) if func_expr.hir_id == child_id => false,

        // Makes an ugly suggestion
        // e.g. *x.deref() => *&*x
        ExprKind::Unary(UnOp::Deref, _)
        // Postfix expressions would require parens
        | ExprKind::Match(_, _, MatchSource::TryDesugar | MatchSource::AwaitDesugar)
        | ExprKind::Field(..)
        | ExprKind::Index(..)
        | ExprKind::Err => false,

        ExprKind::Box(..)
        | ExprKind::ConstBlock(..)
        | ExprKind::Array(_)
        | ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Tup(..)
        | ExprKind::Binary(..)
        | ExprKind::Unary(..)
        | ExprKind::Lit(..)
        | ExprKind::Cast(..)
        | ExprKind::Type(..)
        | ExprKind::DropTemps(..)
        | ExprKind::If(..)
        | ExprKind::Loop(..)
        | ExprKind::Match(..)
        | ExprKind::Let(..)
        | ExprKind::Closure(..)
        | ExprKind::Block(..)
        | ExprKind::Assign(..)
        | ExprKind::AssignOp(..)
        | ExprKind::Path(..)
        | ExprKind::AddrOf(..)
        | ExprKind::Break(..)
        | ExprKind::Continue(..)
        | ExprKind::Ret(..)
        | ExprKind::InlineAsm(..)
        | ExprKind::LlvmInlineAsm(..)
        | ExprKind::Struct(..)
        | ExprKind::Repeat(..)
        | ExprKind::Yield(..) => true,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn report(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, state: State, data: StateData) {
    match state {
        State::DerefMethod {
            ty_changed_count,
            is_final_ufcs,
        } => {
            let mut app = Applicability::MachineApplicable;
            let (expr_str, expr_is_macro_call) = snippet_with_context(cx, expr.span, data.span.ctxt(), "..", &mut app);
            let ty = cx.typeck_results().expr_ty(expr);
            let (_, ref_count) = peel_mid_ty_refs(ty);
            let deref_str = if ty_changed_count >= ref_count && ref_count != 0 {
                // a deref call changing &T -> &U requires two deref operators the first time
                // this occurs. One to remove the reference, a second to call the deref impl.
                "*".repeat(ty_changed_count + 1)
            } else {
                "*".repeat(ty_changed_count)
            };
            let addr_of_str = if ty_changed_count < ref_count {
                // Check if a reborrow from &mut T -> &T is required.
                if data.target_mut == Mutability::Not && matches!(ty.kind(), ty::Ref(_, _, Mutability::Mut)) {
                    "&*"
                } else {
                    ""
                }
            } else if data.target_mut == Mutability::Mut {
                "&mut "
            } else {
                "&"
            };

            let expr_str = if !expr_is_macro_call && is_final_ufcs && expr.precedence().order() < PREC_PREFIX {
                format!("({})", expr_str)
            } else {
                expr_str.into_owned()
            };

            span_lint_and_sugg(
                cx,
                EXPLICIT_DEREF_METHODS,
                data.span,
                match data.target_mut {
                    Mutability::Not => "explicit `deref` method call",
                    Mutability::Mut => "explicit `deref_mut` method call",
                },
                "try this",
                format!("{}{}{}", addr_of_str, deref_str, expr_str),
                app,
            );
        },
    }
}
