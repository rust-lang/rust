use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{is_ty_alias, match_def_path, paths};
use hir::def::Res;
use hir::ExprKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for construction on unit struct using `default`.
    ///
    /// ### Why is this bad?
    /// This adds code complexity and an unnecessary function call.
    ///
    /// ### Example
    /// ```rust
    /// # use std::marker::PhantomData;
    /// #[derive(Default)]
    /// struct S<T> {
    ///     _marker: PhantomData<T>
    /// }
    ///
    /// let _: S<i32> = S {
    ///     _marker: PhantomData::default()
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::marker::PhantomData;
    /// struct S<T> {
    ///     _marker: PhantomData<T>
    /// }
    ///
    /// let _: S<i32> = S {
    ///     _marker: PhantomData
    /// };
    /// ```
    #[clippy::version = "1.71.0"]
    pub DEFAULT_CONSTRUCTED_UNIT_STRUCTS,
    complexity,
    "unit structs can be contructed without calling `default`"
}
declare_lint_pass!(DefaultConstructedUnitStructs => [DEFAULT_CONSTRUCTED_UNIT_STRUCTS]);

fn is_alias(ty: hir::Ty<'_>) -> bool {
    if let hir::TyKind::Path(ref qpath) = ty.kind {
        is_ty_alias(qpath)
    } else {
        false
    }
}

impl LateLintPass<'_> for DefaultConstructedUnitStructs {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if_chain!(
            // make sure we have a call to `Default::default`
            if let hir::ExprKind::Call(fn_expr, &[]) = expr.kind;
            if let ExprKind::Path(ref qpath @ hir::QPath::TypeRelative(base, _)) = fn_expr.kind;
            // make sure this isn't a type alias:
            // `<Foo as Bar>::Assoc` cannot be used as a constructor
            if !is_alias(*base);
            if let Res::Def(_, def_id) = cx.qpath_res(qpath, fn_expr.hir_id);
            if match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            // make sure we have a struct with no fields (unit struct)
            if let ty::Adt(def, ..) = cx.typeck_results().expr_ty(expr).kind();
            if def.is_struct();
            if let var @ ty::VariantDef { ctor: Some((hir::def::CtorKind::Const, _)), .. } = def.non_enum_variant();
            if !var.is_field_list_non_exhaustive();
            if !expr.span.from_expansion() && !qpath.span().from_expansion();
            then {
                span_lint_and_sugg(
                    cx,
                    DEFAULT_CONSTRUCTED_UNIT_STRUCTS,
                    expr.span.with_lo(qpath.qself_span().hi()),
                    "use of `default` to create a unit struct",
                    "remove this call to `default`",
                    String::new(),
                    Applicability::MachineApplicable,
                )
            }
        );
    }
}
