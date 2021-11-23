use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_copy;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

#[allow(clippy::module_name_repetitions)]
#[derive(Copy, Clone)]
pub struct UselessVec {
    pub too_large_for_stack: u64,
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `&vec![..]` when using `&[..]` would
    /// be possible.
    ///
    /// ### Why is this bad?
    /// This is less efficient.
    ///
    /// ### Example
    /// ```rust
    /// # fn foo(my_vec: &[u8]) {}
    ///
    /// // Bad
    /// foo(&vec![1, 2]);
    ///
    /// // Good
    /// foo(&[1, 2]);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_VEC,
    perf,
    "useless `vec!`"
}

impl_lint_pass!(UselessVec => [USELESS_VEC]);

impl<'tcx> LateLintPass<'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // search for `&vec![_]` expressions where the adjusted type is `&[_]`
        if_chain! {
            if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty_adjusted(expr).kind();
            if let ty::Slice(..) = ty.kind();
            if let ExprKind::AddrOf(BorrowKind::Ref, mutability, addressee) = expr.kind;
            if let Some(vec_args) = higher::VecArgs::hir(cx, addressee);
            then {
                self.check_vec_macro(cx, &vec_args, mutability, expr.span);
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
                self.check_vec_macro(cx, &vec_args, Mutability::Not, span);
            }
        }
    }
}

impl UselessVec {
    fn check_vec_macro<'tcx>(
        self,
        cx: &LateContext<'tcx>,
        vec_args: &higher::VecArgs<'tcx>,
        mutability: Mutability,
        span: Span,
    ) {
        let mut applicability = Applicability::MachineApplicable;
        let snippet = match *vec_args {
            higher::VecArgs::Repeat(elem, len) => {
                if let Some((Constant::Int(len_constant), _)) = constant(cx, cx.typeck_results(), len) {
                    #[allow(clippy::cast_possible_truncation)]
                    if len_constant as u64 * size_of(cx, elem) > self.too_large_for_stack {
                        return;
                    }

                    match mutability {
                        Mutability::Mut => {
                            format!(
                                "&mut [{}; {}]",
                                snippet_with_applicability(cx, elem.span, "elem", &mut applicability),
                                snippet_with_applicability(cx, len.span, "len", &mut applicability)
                            )
                        },
                        Mutability::Not => {
                            format!(
                                "&[{}; {}]",
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
                    #[allow(clippy::cast_possible_truncation)]
                    if args.len() as u64 * size_of(cx, last) > self.too_large_for_stack {
                        return;
                    }
                    let span = args[0].span.to(last.span);

                    match mutability {
                        Mutability::Mut => {
                            format!(
                                "&mut [{}]",
                                snippet_with_applicability(cx, span, "..", &mut applicability)
                            )
                        },
                        Mutability::Not => {
                            format!("&[{}]", snippet_with_applicability(cx, span, "..", &mut applicability))
                        },
                    }
                } else {
                    match mutability {
                        Mutability::Mut => "&mut []".into(),
                        Mutability::Not => "&[]".into(),
                    }
                }
            },
        };

        span_lint_and_sugg(
            cx,
            USELESS_VEC,
            span,
            "useless use of `vec!`",
            "you can use a slice directly",
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
