use std::ops::ControlFlow;

use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_copy;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, higher, is_trait_method};
use if_chain::if_chain;
use rustc_ast::BindingAnnotation;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
use rustc_span::sym;

#[expect(clippy::module_name_repetitions)]
#[derive(Copy, Clone)]
pub struct UselessVec {
    pub too_large_for_stack: u64,
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `vec![..]` when using `[..]` would
    /// be possible.
    ///
    /// ### Why is this bad?
    /// This is less efficient.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(_x: &[u8]) {}
    ///
    /// foo(&vec![1, 2]);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # fn foo(_x: &[u8]) {}
    /// foo(&[1, 2]);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_VEC,
    perf,
    "useless `vec!`"
}

impl_lint_pass!(UselessVec => [USELESS_VEC]);

fn adjusts_to_slice(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty_adjusted(e).kind() {
        ty.is_slice()
    } else {
        false
    }
}

/// Checks if the given expression is a method call to a `Vec` method
/// that also exists on slices. If this returns true, it means that
/// this expression does not actually require a `Vec` and could just work with an array.
fn is_allowed_vec_method(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    const ALLOWED_METHOD_NAMES: &[&str] = &["len", "as_ptr", "as_slice", "is_empty"];

    if let ExprKind::MethodCall(path, ..) = e.kind {
        ALLOWED_METHOD_NAMES.contains(&path.ident.name.as_str())
    } else {
        is_trait_method(cx, e, sym::IntoIterator)
    }
}

impl<'tcx> LateLintPass<'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // search for `&vec![_]` expressions where the adjusted type is `&[_]`
        if_chain! {
            if adjusts_to_slice(cx, expr);
            if let ExprKind::AddrOf(BorrowKind::Ref, mutability, addressee) = expr.kind;
            if let Some(vec_args) = higher::VecArgs::hir(cx, addressee);
            then {
                self.check_vec_macro(cx, &vec_args, mutability, expr.span, SuggestSlice::Yes);
            }
        }

        // search for `let foo = vec![_]` expressions where all uses of `foo`
        // adjust to slices or call a method that exist on slices (e.g. len)
        if let Some(vec_args) = higher::VecArgs::hir(cx, expr)
            && let Node::Local(local) = cx.tcx.hir().get_parent(expr.hir_id)
            // for now ignore locals with type annotations.
            // this is to avoid compile errors when doing the suggestion here: let _: Vec<_> = vec![..];
            && local.ty.is_none()
            && let PatKind::Binding(BindingAnnotation(_, mutbl), id, ..) = local.pat.kind
            && is_copy(cx, vec_type(cx.typeck_results().expr_ty_adjusted(expr)))
        {
            let only_slice_uses = for_each_local_use_after_expr(cx, id, expr.hir_id, |expr| {
                // allow indexing into a vec and some set of allowed method calls that exist on slices, too
                if let Some(parent) = get_parent_expr(cx, expr)
                    && (adjusts_to_slice(cx, expr)
                        || matches!(parent.kind, ExprKind::Index(..))
                        || is_allowed_vec_method(cx, parent))
                {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            }).is_continue();

            if only_slice_uses {
                self.check_vec_macro(
                    cx,
                    &vec_args,
                    mutbl,
                    expr.span.ctxt().outer_expn_data().call_site,
                    SuggestSlice::No
                );
            }
        }

        // search for `for _ in vec![…]`
        if_chain! {
            if let Some(higher::ForLoop { arg, .. }) = higher::ForLoop::hir(expr);
            if let Some(vec_args) = higher::VecArgs::hir(cx, arg);
            if is_copy(cx, vec_type(cx.typeck_results().expr_ty_adjusted(arg)));
            then {
                // report the error around the `vec!` not inside `<std macros>:`
                let span = arg.span.ctxt().outer_expn_data().call_site;
                self.check_vec_macro(cx, &vec_args, Mutability::Not, span, SuggestSlice::Yes);
            }
        }
    }
}

#[derive(Copy, Clone)]
enum SuggestSlice {
    /// Suggest using a slice `&[..]` / `&mut [..]`
    Yes,
    /// Suggest using an array: `[..]`
    No,
}

impl UselessVec {
    fn check_vec_macro<'tcx>(
        self,
        cx: &LateContext<'tcx>,
        vec_args: &higher::VecArgs<'tcx>,
        mutability: Mutability,
        span: Span,
        suggest_slice: SuggestSlice,
    ) {
        let mut applicability = Applicability::MachineApplicable;

        let (borrow_prefix_mut, borrow_prefix) = match suggest_slice {
            SuggestSlice::Yes => ("&mut ", "&"),
            SuggestSlice::No => ("", ""),
        };

        let snippet = match *vec_args {
            higher::VecArgs::Repeat(elem, len) => {
                if let Some(Constant::Int(len_constant)) = constant(cx, cx.typeck_results(), len) {
                    #[expect(clippy::cast_possible_truncation)]
                    if len_constant as u64 * size_of(cx, elem) > self.too_large_for_stack {
                        return;
                    }

                    match mutability {
                        Mutability::Mut => {
                            format!(
                                "{borrow_prefix_mut}[{}; {}]",
                                snippet_with_applicability(cx, elem.span, "elem", &mut applicability),
                                snippet_with_applicability(cx, len.span, "len", &mut applicability)
                            )
                        },
                        Mutability::Not => {
                            format!(
                                "{borrow_prefix}[{}; {}]",
                                snippet_with_applicability(cx, elem.span, "elem", &mut applicability),
                                snippet_with_applicability(cx, len.span, "len", &mut applicability)
                            )
                        },
                    }
                } else {
                    return;
                }
            },
            higher::VecArgs::Vec(args) => {
                if let Some(last) = args.iter().last() {
                    if args.len() as u64 * size_of(cx, last) > self.too_large_for_stack {
                        return;
                    }
                    let span = args[0].span.to(last.span);

                    match mutability {
                        Mutability::Mut => {
                            format!(
                                "{borrow_prefix_mut}[{}]",
                                snippet_with_applicability(cx, span, "..", &mut applicability)
                            )
                        },
                        Mutability::Not => {
                            format!(
                                "{borrow_prefix}[{}]",
                                snippet_with_applicability(cx, span, "..", &mut applicability)
                            )
                        },
                    }
                } else {
                    match mutability {
                        Mutability::Mut => format!("{borrow_prefix_mut}[]"),
                        Mutability::Not => format!("{borrow_prefix}[]"),
                    }
                }
            },
        };

        span_lint_and_sugg(
            cx,
            USELESS_VEC,
            span,
            "useless use of `vec!`",
            &format!(
                "you can use {} directly",
                match suggest_slice {
                    SuggestSlice::Yes => "a slice",
                    SuggestSlice::No => "an array",
                }
            ),
            snippet,
            applicability,
        );
    }
}

fn size_of(cx: &LateContext<'_>, expr: &Expr<'_>) -> u64 {
    let ty = cx.typeck_results().expr_ty_adjusted(expr);
    cx.layout_of(ty).map_or(0, |l| l.size.bytes())
}

/// Returns the item type of the vector (i.e., the `T` in `Vec<T>`).
fn vec_type(ty: Ty<'_>) -> Ty<'_> {
    if let ty::Adt(_, substs) = ty.kind() {
        substs.type_at(0)
    } else {
        panic!("The type of `vec!` is a not a struct?");
    }
}
