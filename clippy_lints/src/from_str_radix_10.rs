use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::span_lint_and_sugg;

declare_clippy_lint! {
    /// **What it does:**
    /// Checks for function invocations of the form `primitive::from_str_radix(s, 10)`
    ///
    /// **Why is this bad?**
    /// This specific common use case can be rewritten as `s.parse::<primitive>()`
    /// (and in most cases, the turbofish can be removed), which reduces code length
    /// and complexity.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let input: &str = get_input();
    /// let num = u16::from_str_radix(input, 10)?;
    /// ```
    /// Use instead:
    /// ```rust
    /// let input: &str = get_input();
    /// let num: u16 = input.parse()?;
    /// ```
    pub FROM_STR_RADIX_10,
    style,
    "from_str_radix with radix 10"
}

declare_lint_pass!(FromStrRadix10 => [FROM_STR_RADIX_10]);

impl LateLintPass<'tcx> for FromStrRadix10 {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, exp: &Expr<'tcx>) {
        if_chain! {
            if let ExprKind::Call(maybe_path, arguments) = &exp.kind;
            if let ExprKind::Path(qpath) = &maybe_path.kind;
            if let QPath::TypeRelative(ty, pathseg) = &qpath;

            // check if the first part of the path is some integer primitive
            if let TyKind::Path(ty_qpath) = &ty.kind;
            let ty_res = cx.qpath_res(ty_qpath, ty.hir_id);
            if let def::Res::PrimTy(prim_ty) = ty_res;
            if is_primitive_integer_ty(prim_ty);

            // check if the second part of the path indeed calls the associated
            // function `from_str_radix`
            if pathseg.ident.name.as_str() == "from_str_radix";

            // check if the second argument is a primitive `10`
            if arguments.len() == 2;
            if let ExprKind::Lit(lit) = &arguments[1].kind;
            if let rustc_ast::ast::LitKind::Int(10, _) = lit.node;

            then {
                let orig_string = crate::utils::snippet(cx, arguments[0].span, "string");
                span_lint_and_sugg(
                    cx,
                    FROM_STR_RADIX_10,
                    exp.span,
                    "This call to `from_str_radix` can be shortened to a call to str::parse",
                    "try",
                    format!("({}).parse()", orig_string),
                    Applicability::MaybeIncorrect
                );
            }
        }
    }
}

fn is_primitive_integer_ty(ty: PrimTy) -> bool {
    match ty {
        PrimTy::Int(_) => true,
        PrimTy::Uint(_) => true,
        _ => false,
    }
}
