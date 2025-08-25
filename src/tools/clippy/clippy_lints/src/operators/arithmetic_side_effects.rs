use super::ARITHMETIC_SIDE_EFFECTS;
use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{expr_or_init, is_from_proc_macro, is_lint_allowed, peel_hir_expr_refs, peel_hir_expr_unary, sym};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};
use {rustc_ast as ast, rustc_hir as hir};

pub struct ArithmeticSideEffects {
    allowed_binary: FxHashMap<&'static str, FxHashSet<&'static str>>,
    allowed_unary: FxHashSet<&'static str>,
    // Used to check whether expressions are constants, such as in enum discriminants and consts
    const_span: Option<Span>,
    disallowed_int_methods: FxHashSet<Symbol>,
    expr_span: Option<Span>,
}

impl_lint_pass!(ArithmeticSideEffects => [ARITHMETIC_SIDE_EFFECTS]);

impl ArithmeticSideEffects {
    pub fn new(conf: &'static Conf) -> Self {
        let mut allowed_binary = FxHashMap::<&'static str, FxHashSet<&'static str>>::default();
        let mut allowed_unary = FxHashSet::<&'static str>::default();

        allowed_unary.extend(["f32", "f64", "std::num::Saturating", "std::num::Wrapping"]);
        allowed_unary.extend(conf.arithmetic_side_effects_allowed_unary.iter().map(|x| &**x));
        allowed_binary.extend([
            ("f32", FxHashSet::from_iter(["f32"])),
            ("f64", FxHashSet::from_iter(["f64"])),
            ("std::string::String", FxHashSet::from_iter(["str"])),
        ]);
        for (lhs, rhs) in &conf.arithmetic_side_effects_allowed_binary {
            allowed_binary.entry(lhs).or_default().insert(rhs);
        }
        for s in &conf.arithmetic_side_effects_allowed {
            allowed_binary.entry(s).or_default().insert("*");
            allowed_binary.entry("*").or_default().insert(s);
            allowed_unary.insert(s);
        }

        Self {
            allowed_binary,
            allowed_unary,
            const_span: None,
            disallowed_int_methods: [
                sym::saturating_div,
                sym::wrapping_div,
                sym::wrapping_rem,
                sym::wrapping_rem_euclid,
            ]
            .into_iter()
            .collect(),
            expr_span: None,
        }
    }

    /// Checks if the lhs and the rhs types of a binary operation like "addition" or
    /// "multiplication" are present in the inner set of allowed types.
    fn has_allowed_binary(&self, lhs_ty: Ty<'_>, rhs_ty: Ty<'_>) -> bool {
        let lhs_ty_string = lhs_ty.to_string();
        let lhs_ty_string_elem = lhs_ty_string.split('<').next().unwrap_or_default();
        let rhs_ty_string = rhs_ty.to_string();
        let rhs_ty_string_elem = rhs_ty_string.split('<').next().unwrap_or_default();
        if let Some(rhs_from_specific) = self.allowed_binary.get(lhs_ty_string_elem)
            && {
                let rhs_has_allowed_ty = rhs_from_specific.contains(rhs_ty_string_elem);
                rhs_has_allowed_ty || rhs_from_specific.contains("*")
            }
        {
            true
        } else if let Some(rhs_from_glob) = self.allowed_binary.get("*") {
            rhs_from_glob.contains(rhs_ty_string_elem)
        } else {
            false
        }
    }

    /// Checks if the type of an unary operation like "negation" is present in the inner set of
    /// allowed types.
    fn has_allowed_unary(&self, ty: Ty<'_>) -> bool {
        let ty_string = ty.to_string();
        let ty_string_elem = ty_string.split('<').next().unwrap_or_default();
        self.allowed_unary.contains(ty_string_elem)
    }

    fn is_non_zero_u(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
        if let ty::Adt(adt, substs) = ty.kind()
            && cx.tcx.is_diagnostic_item(sym::NonZero, adt.did())
            && let int_type = substs.type_at(0)
            && matches!(int_type.kind(), ty::Uint(_))
        {
            true
        } else {
            false
        }
    }

    /// Verifies built-in types that have specific allowed operations
    fn has_specific_allowed_type_and_operation<'tcx>(
        cx: &LateContext<'tcx>,
        lhs_ty: Ty<'tcx>,
        op: hir::BinOpKind,
        rhs_ty: Ty<'tcx>,
    ) -> bool {
        let is_div_or_rem = matches!(op, hir::BinOpKind::Div | hir::BinOpKind::Rem);
        let is_sat_or_wrap = |ty: Ty<'_>| {
            is_type_diagnostic_item(cx, ty, sym::Saturating) || is_type_diagnostic_item(cx, ty, sym::Wrapping)
        };

        // If the RHS is `NonZero<u*>`, then division or module by zero will never occur.
        if Self::is_non_zero_u(cx, rhs_ty) && is_div_or_rem {
            return true;
        }

        // `Saturation` and `Wrapping` can overflow if the RHS is zero in a division or module.
        if is_sat_or_wrap(lhs_ty) {
            return !is_div_or_rem;
        }

        false
    }

    // For example, 8i32 or &i64::MAX.
    fn is_integral(ty: Ty<'_>) -> bool {
        ty.peel_refs().is_integral()
    }

    // Common entry-point to avoid code duplication.
    fn issue_lint<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if is_from_proc_macro(cx, expr) {
            return;
        }

        let msg = "arithmetic operation that can potentially result in unexpected side-effects";
        span_lint(cx, ARITHMETIC_SIDE_EFFECTS, expr.span, msg);
        self.expr_span = Some(expr.span);
    }

    /// Returns the numeric value of a literal integer originated from `expr`, if any.
    ///
    /// Literal integers can be originated from adhoc declarations like `1`, associated constants
    /// like `i32::MAX` or constant references like `N` from `const N: i32 = 1;`,
    fn literal_integer(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<u128> {
        let actual = peel_hir_expr_unary(expr).0;
        if let hir::ExprKind::Lit(lit) = actual.kind
            && let ast::LitKind::Int(n, _) = lit.node
        {
            return Some(n.get());
        }
        if let Some(Constant::Int(n)) = ConstEvalCtxt::new(cx).eval(expr) {
            return Some(n);
        }
        None
    }

    /// Methods like `add_assign` are send to their `BinOps` references.
    fn manage_sugar_methods<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        lhs: &'tcx hir::Expr<'_>,
        ps: &hir::PathSegment<'_>,
        rhs: &'tcx hir::Expr<'_>,
    ) {
        if ps.ident.name == sym::add || ps.ident.name == sym::add_assign {
            self.manage_bin_ops(cx, expr, hir::BinOpKind::Add, lhs, rhs);
        } else if ps.ident.name == sym::div || ps.ident.name == sym::div_assign {
            self.manage_bin_ops(cx, expr, hir::BinOpKind::Div, lhs, rhs);
        } else if ps.ident.name == sym::mul || ps.ident.name == sym::mul_assign {
            self.manage_bin_ops(cx, expr, hir::BinOpKind::Mul, lhs, rhs);
        } else if ps.ident.name == sym::rem || ps.ident.name == sym::rem_assign {
            self.manage_bin_ops(cx, expr, hir::BinOpKind::Rem, lhs, rhs);
        } else if ps.ident.name == sym::sub || ps.ident.name == sym::sub_assign {
            self.manage_bin_ops(cx, expr, hir::BinOpKind::Sub, lhs, rhs);
        }
    }

    /// Manages when the lint should be triggered. Operations in constant environments, hard coded
    /// types, custom allowed types and non-constant operations that don't overflow are ignored.
    fn manage_bin_ops<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        op: hir::BinOpKind,
        lhs: &'tcx hir::Expr<'_>,
        rhs: &'tcx hir::Expr<'_>,
    ) {
        if ConstEvalCtxt::new(cx).eval_simple(expr).is_some() {
            return;
        }
        if !matches!(
            op,
            hir::BinOpKind::Add
                | hir::BinOpKind::Div
                | hir::BinOpKind::Mul
                | hir::BinOpKind::Rem
                | hir::BinOpKind::Shl
                | hir::BinOpKind::Shr
                | hir::BinOpKind::Sub
        ) {
            return;
        }
        let (mut actual_lhs, lhs_ref_counter) = peel_hir_expr_refs(lhs);
        let (mut actual_rhs, rhs_ref_counter) = peel_hir_expr_refs(rhs);
        actual_lhs = expr_or_init(cx, actual_lhs);
        actual_rhs = expr_or_init(cx, actual_rhs);

        // `NonZeroU*.get() - 1`, will never overflow
        if let hir::BinOpKind::Sub = op
            && let hir::ExprKind::MethodCall(method, receiver, [], _) = actual_lhs.kind
            && method.ident.name == sym::get
            && let receiver_ty = cx.typeck_results().expr_ty(receiver).peel_refs()
            && Self::is_non_zero_u(cx, receiver_ty)
            && let Some(1) = Self::literal_integer(cx, actual_rhs)
        {
            return;
        }

        let lhs_ty = cx.typeck_results().expr_ty(actual_lhs).peel_refs();
        let rhs_ty = cx.typeck_results().expr_ty_adjusted(actual_rhs).peel_refs();
        if self.has_allowed_binary(lhs_ty, rhs_ty) {
            return;
        }
        if Self::has_specific_allowed_type_and_operation(cx, lhs_ty, op, rhs_ty) {
            return;
        }

        let has_valid_op = if Self::is_integral(lhs_ty) && Self::is_integral(rhs_ty) {
            if let hir::BinOpKind::Shl | hir::BinOpKind::Shr = op {
                // At least for integers, shifts are already handled by the CTFE
                return;
            }
            match (
                Self::literal_integer(cx, actual_lhs),
                Self::literal_integer(cx, actual_rhs),
            ) {
                (None, None) => false,
                (None, Some(n)) => match (&op, n) {
                    // Division and module are always valid if applied to non-zero integers
                    (hir::BinOpKind::Div | hir::BinOpKind::Rem, local_n) if local_n != 0 => true,
                    // Adding or subtracting zeros is always a no-op
                    (hir::BinOpKind::Add | hir::BinOpKind::Sub, 0)
                    // Multiplication by 1 or 0 will never overflow
                    | (hir::BinOpKind::Mul, 0 | 1)
                    => true,
                    _ => false,
                },
                (Some(n), None) => match (&op, n) {
                    // Adding or subtracting zeros is always a no-op
                    (hir::BinOpKind::Add | hir::BinOpKind::Sub, 0)
                    // Multiplication by 1 or 0 will never overflow
                    | (hir::BinOpKind::Mul, 0 | 1)
                    => true,
                    _ => false,
                },
                (Some(_), Some(_)) => {
                    matches!((lhs_ref_counter, rhs_ref_counter), (0, 0))
                },
            }
        } else {
            false
        };
        if !has_valid_op {
            self.issue_lint(cx, expr);
        }
    }

    /// There are some integer methods like `wrapping_div` that will panic depending on the
    /// provided input.
    fn manage_method_call<'tcx>(
        &mut self,
        args: &'tcx [hir::Expr<'_>],
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        ps: &'tcx hir::PathSegment<'_>,
        receiver: &'tcx hir::Expr<'_>,
    ) {
        let Some(arg) = args.first() else {
            return;
        };
        if ConstEvalCtxt::new(cx).eval_simple(receiver).is_some() {
            return;
        }
        let instance_ty = cx.typeck_results().expr_ty_adjusted(receiver);
        if !Self::is_integral(instance_ty) {
            return;
        }
        self.manage_sugar_methods(cx, expr, receiver, ps, arg);
        if !self.disallowed_int_methods.contains(&ps.ident.name) {
            return;
        }
        let (actual_arg, _) = peel_hir_expr_refs(arg);
        match Self::literal_integer(cx, actual_arg) {
            None | Some(0) => self.issue_lint(cx, arg),
            Some(_) => {},
        }
    }

    fn manage_unary_ops<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        un_expr: &'tcx hir::Expr<'_>,
        un_op: hir::UnOp,
    ) {
        let hir::UnOp::Neg = un_op else {
            return;
        };
        if ConstEvalCtxt::new(cx).eval(un_expr).is_some() {
            return;
        }
        let ty = cx.typeck_results().expr_ty_adjusted(expr).peel_refs();
        if self.has_allowed_unary(ty) {
            return;
        }
        let actual_un_expr = peel_hir_expr_refs(un_expr).0;
        if Self::literal_integer(cx, actual_un_expr).is_some() {
            return;
        }
        self.issue_lint(cx, expr);
    }

    fn should_skip_expr<'tcx>(&self, cx: &LateContext<'tcx>, expr: &hir::Expr<'tcx>) -> bool {
        is_lint_allowed(cx, ARITHMETIC_SIDE_EFFECTS, expr.hir_id)
            || self.expr_span.is_some()
            || self.const_span.is_some_and(|sp| sp.contains(expr.span))
    }
}

impl<'tcx> LateLintPass<'tcx> for ArithmeticSideEffects {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if self.should_skip_expr(cx, expr) {
            return;
        }
        match &expr.kind {
            hir::ExprKind::Binary(op, lhs, rhs) => {
                self.manage_bin_ops(cx, expr, op.node, lhs, rhs);
            },
            hir::ExprKind::AssignOp(op, lhs, rhs) => {
                self.manage_bin_ops(cx, expr, op.node.into(), lhs, rhs);
            },
            hir::ExprKind::MethodCall(ps, receiver, args, _) => {
                self.manage_method_call(args, cx, expr, ps, receiver);
            },
            hir::ExprKind::Unary(un_op, un_expr) => {
                self.manage_unary_ops(cx, expr, un_expr, *un_op);
            },
            _ => {},
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir_body_owner(body.id());
        let body_owner_def_id = cx.tcx.hir_body_owner_def_id(body.id());

        let body_owner_kind = cx.tcx.hir_body_owner_kind(body_owner_def_id);
        if let hir::BodyOwnerKind::Const { .. } | hir::BodyOwnerKind::Static(_) = body_owner_kind {
            let body_span = cx.tcx.hir_span_with_body(body_owner);
            if let Some(span) = self.const_span
                && span.contains(body_span)
            {
                return;
            }
            self.const_span = Some(body_span);
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir_body_owner(body.id());
        let body_span = cx.tcx.hir_span(body_owner);
        if let Some(span) = self.const_span
            && span.contains(body_span)
        {
            return;
        }
        self.const_span = None;
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if Some(expr.span) == self.expr_span {
            self.expr_span = None;
        }
    }
}
