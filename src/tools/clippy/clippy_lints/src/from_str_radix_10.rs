use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef as _;
use clippy_utils::source::snippet_opt;
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_in_const_context, is_integer_literal, sym};
use rustc_attr_ir::lang_items::LangItem;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind, PrimTy, QPath, TyKind, def};
use rustc_lint::{LateContext, LateLintPass, impl_lint_pass};
use rustc_middle::ty::{self, Ty};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for function invocations of the form `integer::from_str_radix(s, 10)`
    ///
    /// This also applies to `NonZero` integer types on Rust 1.98 and later.
    ///
    /// ### Why is this bad?
    ///
    /// This specific common use case can be rewritten as `s.parse::<integer>()`
    /// (and in most cases, the turbofish can be removed), which reduces code length
    /// and complexity.
    ///
    /// ### Known problems
    ///
    /// This lint may suggest using `(&<expression>).parse()` instead of `<expression>.parse()`
    /// directly in some cases, which is correct but adds unnecessary complexity to the code.
    ///
    /// ### Example
    /// ```ignore
    /// let input: &str = get_input();
    /// let num = u16::from_str_radix(input, 10)?;
    /// ```
    /// Use instead:
    /// ```ignore
    /// let input: &str = get_input();
    /// let num: u16 = input.parse()?;
    /// ```
    #[clippy::version = "1.52.0"]
    pub FROM_STR_RADIX_10,
    style,
    "from_str_radix with radix 10"
}

impl_lint_pass!(FromStrRadix10 => [FROM_STR_RADIX_10]);

pub struct FromStrRadix10 {
    msrv: Msrv,
}

impl FromStrRadix10 {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv.into() }
    }
}

impl<'tcx> LateLintPass<'tcx> for FromStrRadix10 {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, exp: &Expr<'tcx>) {
        if exp.span.from_expansion() {
            return;
        }

        if let ExprKind::Call(maybe_path, [src, radix]) = &exp.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, pathseg)) = &maybe_path.kind

            // check if the second argument is a primitive `10`
            && is_integer_literal(radix, 10)

            // check if the second part of the path indeed calls the associated
            // function `from_str_radix`
            && pathseg.ident.name == sym::from_str_radix

            // check if the first part of the path is some integer primitive or NonZero integer
            && let TyKind::Path(ty_qpath) = &ty.kind
            && let Some(integer_ty) = get_integer_ty(cx, ty, ty_qpath)
            && (matches!(integer_ty, IntegerTy::Primitive(_)) || self.msrv.meets(cx, msrvs::NONZERO_FROM_STR_RADIX))

            // do not lint in constant context, because the suggestion won't work.
            // NB: keep this check until a new `const_trait_impl` is available and stabilized.
            && !is_in_const_context(cx)
        {
            let expr = if let ExprKind::AddrOf(_, _, expr) = &src.kind {
                let ty = cx.typeck_results().expr_ty(expr);
                if is_ty_stringish(cx, ty) { expr } else { &src }
            } else {
                &src
            };

            let type_name = match integer_ty {
                IntegerTy::Primitive(prim_ty) => prim_ty.name_str().to_owned(),
                IntegerTy::NonZero(ty) => {
                    // Keep the source spelling, including aliases like `MyNonZero` and
                    // qualified paths like `std::num::NonZeroU8`.
                    let Some(type_name) = snippet_opt(cx, ty.span) else {
                        return;
                    };
                    type_name
                },
            };
            let sugg =
                Sugg::hir_with_applicability(cx, expr, "<string>", &mut Applicability::MachineApplicable).maybe_paren();

            span_lint_and_sugg(
                cx,
                FROM_STR_RADIX_10,
                exp.span,
                "this call to `from_str_radix` can be replaced with a call to `str::parse`",
                "try",
                format!("{sugg}.parse::<{type_name}>()"),
                Applicability::MaybeIncorrect,
            );
        }
    }
}

enum IntegerTy<'tcx> {
    Primitive(PrimTy),
    NonZero(&'tcx hir::Ty<'tcx>),
}

fn get_integer_ty<'tcx>(
    cx: &LateContext<'tcx>,
    ty: &'tcx hir::Ty<'tcx>,
    ty_qpath: &QPath<'tcx>,
) -> Option<IntegerTy<'tcx>> {
    match cx.qpath_res(ty_qpath, ty.hir_id) {
        def::Res::PrimTy(prim_ty) if matches!(prim_ty, PrimTy::Int(_) | PrimTy::Uint(_)) => {
            Some(IntegerTy::Primitive(prim_ty))
        },
        _ if let ty::Adt(adt, args) = cx.typeck_results().node_type(ty.hir_id).kind()
            && cx.tcx.is_diagnostic_item(sym::NonZero, adt.did())
            && matches!(args.type_at(0).kind(), ty::Int(_) | ty::Uint(_)) =>
        {
            Some(IntegerTy::NonZero(ty))
        },
        _ => None,
    }
}

/// Checks if a Ty is `String` or `&str`
fn is_ty_stringish(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    ty.is_lang_item(cx, LangItem::String) || ty.peel_refs().is_str()
}
