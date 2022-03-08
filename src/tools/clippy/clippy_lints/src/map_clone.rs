use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item};
use clippy_utils::{is_trait_method, meets_msrv, msrvs, peel_blocks};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::Mutability;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `map(|x| x.clone())` or
    /// dereferencing closures for `Copy` types, on `Iterator` or `Option`,
    /// and suggests `cloned()` or `copied()` instead
    ///
    /// ### Why is this bad?
    /// Readability, this can be written more concisely
    ///
    /// ### Example
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.map(|i| *i);
    /// ```
    ///
    /// The correct use would be:
    ///
    /// ```rust
    /// let x = vec![42, 43];
    /// let y = x.iter();
    /// let z = y.cloned();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MAP_CLONE,
    style,
    "using `iterator.map(|x| x.clone())`, or dereferencing closures for `Copy` types"
}

pub struct MapClone {
    msrv: Option<RustcVersion>,
}

impl_lint_pass!(MapClone => [MAP_CLONE]);

impl MapClone {
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for MapClone {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &hir::Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if_chain! {
            if let hir::ExprKind::MethodCall(method, args, _) = e.kind;
            if args.len() == 2;
            if method.ident.name == sym::map;
            let ty = cx.typeck_results().expr_ty(&args[0]);
            if is_type_diagnostic_item(cx, ty, sym::Option) || is_trait_method(cx, e, sym::Iterator);
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = args[1].kind;
            then {
                let closure_body = cx.tcx.hir().body(body_id);
                let closure_expr = peel_blocks(&closure_body.value);
                match closure_body.params[0].pat.kind {
                    hir::PatKind::Ref(inner, hir::Mutability::Not) => if let hir::PatKind::Binding(
                        hir::BindingAnnotation::Unannotated, .., name, None
                    ) = inner.kind {
                        if ident_eq(name, closure_expr) {
                            self.lint_explicit_closure(cx, e.span, args[0].span, true);
                        }
                    },
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, .., name, None) => {
                        match closure_expr.kind {
                            hir::ExprKind::Unary(hir::UnOp::Deref, inner) => {
                                if ident_eq(name, inner) {
                                    if let ty::Ref(.., Mutability::Not) = cx.typeck_results().expr_ty(inner).kind() {
                                        self.lint_explicit_closure(cx, e.span, args[0].span, true);
                                    }
                                }
                            },
                            hir::ExprKind::MethodCall(method, [obj], _) => if_chain! {
                                if ident_eq(name, obj) && method.ident.name == sym::clone;
                                if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id);
                                if let Some(trait_id) = cx.tcx.trait_of_item(fn_id);
                                if cx.tcx.lang_items().clone_trait().map_or(false, |id| id == trait_id);
                                // no autoderefs
                                if !cx.typeck_results().expr_adjustments(obj).iter()
                                    .any(|a| matches!(a.kind, Adjust::Deref(Some(..))));
                                then {
                                    let obj_ty = cx.typeck_results().expr_ty(obj);
                                    if let ty::Ref(_, ty, mutability) = obj_ty.kind() {
                                        if matches!(mutability, Mutability::Not) {
                                            let copy = is_copy(cx, *ty);
                                            self.lint_explicit_closure(cx, e.span, args[0].span, copy);
                                        }
                                    } else {
                                        lint_needless_cloning(cx, e.span, args[0].span);
                                    }
                                }
                            },
                            _ => {},
                        }
                    },
                    _ => {},
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn ident_eq(name: Ident, path: &hir::Expr<'_>) -> bool {
    if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = path.kind {
        path.segments.len() == 1 && path.segments[0].ident == name
    } else {
        false
    }
}

fn lint_needless_cloning(cx: &LateContext<'_>, root: Span, receiver: Span) {
    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        root.trim_start(receiver).unwrap(),
        "you are needlessly cloning iterator elements",
        "remove the `map` call",
        String::new(),
        Applicability::MachineApplicable,
    );
}

impl MapClone {
    fn lint_explicit_closure(&self, cx: &LateContext<'_>, replace: Span, root: Span, is_copy: bool) {
        let mut applicability = Applicability::MachineApplicable;
        let message = if is_copy {
            "you are using an explicit closure for copying elements"
        } else {
            "you are using an explicit closure for cloning elements"
        };
        let sugg_method = if is_copy && meets_msrv(self.msrv.as_ref(), &msrvs::ITERATOR_COPIED) {
            "copied"
        } else {
            "cloned"
        };

        span_lint_and_sugg(
            cx,
            MAP_CLONE,
            replace,
            message,
            &format!("consider calling the dedicated `{}` method", sugg_method),
            format!(
                "{}.{}()",
                snippet_with_applicability(cx, root, "..", &mut applicability),
                sugg_method,
            ),
            applicability,
        );
    }
}
