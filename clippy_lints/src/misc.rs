use if_chain::if_chain;
use matches::matches;
use rustc::hir::intravisit::FnKind;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::LitKind;
use syntax::source_map::{ExpnFormat, Span};

use crate::consts::{constant, Constant};
use crate::utils::sugg::Sugg;
use crate::utils::{
    get_item_name, get_parent_expr, implements_trait, in_constant, in_macro_or_desugar, is_integer_literal,
    iter_input_pats, last_path_segment, match_qpath, match_trait_method, paths, snippet, span_lint, span_lint_and_then,
    span_lint_hir_and_then, walk_ptrs_ty, SpanlessEq,
};

declare_clippy_lint! {
    /// **What it does:** Checks for function arguments and let bindings denoted as
    /// `ref`.
    ///
    /// **Why is this bad?** The `ref` declaration makes the function take an owned
    /// value, but turns the argument into a reference (which means that the value
    /// is destroyed when exiting the function). This adds not much value: either
    /// take a reference type, or take an owned value and create references in the
    /// body.
    ///
    /// For let bindings, `let x = &foo;` is preferred over `let ref x = foo`. The
    /// type of `x` is more obvious with the former.
    ///
    /// **Known problems:** If the argument is dereferenced within the function,
    /// removing the `ref` will lead to errors. This can be fixed by removing the
    /// dereferences, e.g., changing `*x` to `x` within the function.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(ref x: u8) -> bool {
    ///     true
    /// }
    /// ```
    pub TOPLEVEL_REF_ARG,
    style,
    "an entire binding declared as `ref`, in a function argument or a `let` statement"
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparisons to NaN.
    ///
    /// **Why is this bad?** NaN does not compare meaningfully to anything – not
    /// even itself – so those comparisons are simply wrong.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use core::f32::NAN;
    /// # let x = 1.0;
    ///
    /// if x == NAN { }
    /// ```
    pub CMP_NAN,
    correctness,
    "comparisons to NAN, which will always return false, probably not intended"
}

declare_clippy_lint! {
    /// **What it does:** Checks for (in-)equality comparisons on floating-point
    /// values (apart from zero), except in functions called `*eq*` (which probably
    /// implement equality for a type involving floats).
    ///
    /// **Why is this bad?** Floating point calculations are usually imprecise, so
    /// asking if two values are *exactly* equal is asking for trouble. For a good
    /// guide on what to do, see [the floating point
    /// guide](http://www.floating-point-gui.de/errors/comparison).
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = 1.2331f64;
    /// let y = 1.2332f64;
    /// if y == 1.23f64 { }
    /// if y != x {} // where both are floats
    /// ```
    pub FLOAT_CMP,
    correctness,
    "using `==` or `!=` on float values instead of comparing difference with an epsilon"
}

declare_clippy_lint! {
    /// **What it does:** Checks for conversions to owned values just for the sake
    /// of a comparison.
    ///
    /// **Why is this bad?** The comparison can operate on a reference, so creating
    /// an owned value effectively throws it away directly afterwards, which is
    /// needlessly consuming code and heap space.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// x.to_owned() == y
    /// ```
    pub CMP_OWNED,
    perf,
    "creating owned instances for comparing with others, e.g., `x == \"foo\".to_string()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for getting the remainder of a division by one.
    ///
    /// **Why is this bad?** The result can only ever be zero. No one will write
    /// such code deliberately, unless trying to win an Underhanded Rust
    /// Contest. Even for that contest, it's probably a bad idea. Use something more
    /// underhanded.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let x = 1;
    /// let a = x % 1;
    /// ```
    pub MODULO_ONE,
    correctness,
    "taking a number modulo 1, which always returns 0"
}

declare_clippy_lint! {
    /// **What it does:** Checks for patterns in the form `name @ _`.
    ///
    /// **Why is this bad?** It's almost always more readable to just use direct
    /// bindings.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let v = Some("abc");
    ///
    /// match v {
    ///     Some(x) => (),
    ///     y @ _ => (), // easier written as `y`,
    /// }
    /// ```
    pub REDUNDANT_PATTERN,
    style,
    "using `name @ _` in a pattern"
}

declare_clippy_lint! {
    /// **What it does:** Checks for the use of bindings with a single leading
    /// underscore.
    ///
    /// **Why is this bad?** A single leading underscore is usually used to indicate
    /// that a binding will not be used. Using such a binding breaks this
    /// expectation.
    ///
    /// **Known problems:** The lint does not work properly with desugaring and
    /// macro, it has been allowed in the mean time.
    ///
    /// **Example:**
    /// ```rust
    /// let _x = 0;
    /// let y = _x + 1; // Here we are using `_x`, even though it has a leading
    ///                 // underscore. We should rename `_x` to `x`
    /// ```
    pub USED_UNDERSCORE_BINDING,
    pedantic,
    "using a binding which is prefixed with an underscore"
}

declare_clippy_lint! {
    /// **What it does:** Checks for the use of short circuit boolean conditions as
    /// a
    /// statement.
    ///
    /// **Why is this bad?** Using a short circuit boolean condition as a statement
    /// may hide the fact that the second part is executed or not depending on the
    /// outcome of the first part.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// f() && g(); // We should write `if f() { g(); }`.
    /// ```
    pub SHORT_CIRCUIT_STATEMENT,
    complexity,
    "using a short circuit boolean condition as a statement"
}

declare_clippy_lint! {
    /// **What it does:** Catch casts from `0` to some pointer type
    ///
    /// **Why is this bad?** This generally means `null` and is better expressed as
    /// {`std`, `core`}`::ptr::`{`null`, `null_mut`}.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let a = 0 as *const u32;
    /// ```
    pub ZERO_PTR,
    style,
    "using 0 as *{const, mut} T"
}

declare_clippy_lint! {
    /// **What it does:** Checks for (in-)equality comparisons on floating-point
    /// value and constant, except in functions called `*eq*` (which probably
    /// implement equality for a type involving floats).
    ///
    /// **Why is this bad?** Floating point calculations are usually imprecise, so
    /// asking if two values are *exactly* equal is asking for trouble. For a good
    /// guide on what to do, see [the floating point
    /// guide](http://www.floating-point-gui.de/errors/comparison).
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// const ONE = 1.00f64;
    /// x == ONE  // where both are floats
    /// ```
    pub FLOAT_CMP_CONST,
    restriction,
    "using `==` or `!=` on float constants instead of comparing difference with an epsilon"
}

declare_lint_pass!(MiscLints => [
    TOPLEVEL_REF_ARG,
    CMP_NAN,
    FLOAT_CMP,
    CMP_OWNED,
    MODULO_ONE,
    REDUNDANT_PATTERN,
    USED_UNDERSCORE_BINDING,
    SHORT_CIRCUIT_STATEMENT,
    ZERO_PTR,
    FLOAT_CMP_CONST
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MiscLints {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        k: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        _: HirId,
    ) {
        if let FnKind::Closure(_) = k {
            // Does not apply to closures
            return;
        }
        for arg in iter_input_pats(decl, body) {
            match arg.pat.node {
                PatKind::Binding(BindingAnnotation::Ref, ..) | PatKind::Binding(BindingAnnotation::RefMut, ..) => {
                    span_lint(
                        cx,
                        TOPLEVEL_REF_ARG,
                        arg.pat.span,
                        "`ref` directly on a function argument is ignored. Consider using a reference type \
                         instead.",
                    );
                },
                _ => {},
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, s: &'tcx Stmt) {
        if_chain! {
            if let StmtKind::Local(ref l) = s.node;
            if let PatKind::Binding(an, .., i, None) = l.pat.node;
            if let Some(ref init) = l.init;
            then {
                if an == BindingAnnotation::Ref || an == BindingAnnotation::RefMut {
                    let sugg_init = Sugg::hir(cx, init, "..");
                    let (mutopt,initref) = if an == BindingAnnotation::RefMut {
                        ("mut ", sugg_init.mut_addr())
                    } else {
                        ("", sugg_init.addr())
                    };
                    let tyopt = if let Some(ref ty) = l.ty {
                        format!(": &{mutopt}{ty}", mutopt=mutopt, ty=snippet(cx, ty.span, "_"))
                    } else {
                        String::new()
                    };
                    span_lint_hir_and_then(cx,
                        TOPLEVEL_REF_ARG,
                        init.hir_id,
                        l.pat.span,
                        "`ref` on an entire `let` pattern is discouraged, take a reference with `&` instead",
                        |db| {
                            db.span_suggestion(
                                s.span,
                                "try",
                                format!(
                                    "let {name}{tyopt} = {initref};",
                                    name=snippet(cx, i.span, "_"),
                                    tyopt=tyopt,
                                    initref=initref,
                                ),
                                Applicability::MachineApplicable, // snippet
                            );
                        }
                    );
                }
            }
        };
        if_chain! {
            if let StmtKind::Semi(ref expr) = s.node;
            if let ExprKind::Binary(ref binop, ref a, ref b) = expr.node;
            if binop.node == BinOpKind::And || binop.node == BinOpKind::Or;
            if let Some(sugg) = Sugg::hir_opt(cx, a);
            then {
                span_lint_and_then(cx,
                    SHORT_CIRCUIT_STATEMENT,
                    s.span,
                    "boolean short circuit operator in statement may be clearer using an explicit test",
                    |db| {
                        let sugg = if binop.node == BinOpKind::Or { !sugg } else { sugg };
                        db.span_suggestion(
                            s.span,
                            "replace it with",
                            format!(
                                "if {} {{ {}; }}",
                                sugg,
                                &snippet(cx, b.span, ".."),
                            ),
                            Applicability::MachineApplicable, // snippet
                        );
                    });
            }
        };
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        match expr.node {
            ExprKind::Cast(ref e, ref ty) => {
                check_cast(cx, expr.span, e, ty);
                return;
            },
            ExprKind::Binary(ref cmp, ref left, ref right) => {
                let op = cmp.node;
                if op.is_comparison() {
                    if let ExprKind::Path(QPath::Resolved(_, ref path)) = left.node {
                        check_nan(cx, path, expr);
                    }
                    if let ExprKind::Path(QPath::Resolved(_, ref path)) = right.node {
                        check_nan(cx, path, expr);
                    }
                    check_to_owned(cx, left, right);
                    check_to_owned(cx, right, left);
                }
                if (op == BinOpKind::Eq || op == BinOpKind::Ne) && (is_float(cx, left) || is_float(cx, right)) {
                    if is_allowed(cx, left) || is_allowed(cx, right) {
                        return;
                    }
                    if let Some(name) = get_item_name(cx, expr) {
                        let name = name.as_str();
                        if name == "eq"
                            || name == "ne"
                            || name == "is_nan"
                            || name.starts_with("eq_")
                            || name.ends_with("_eq")
                        {
                            return;
                        }
                    }
                    let (lint, msg) = if is_named_constant(cx, left) || is_named_constant(cx, right) {
                        (FLOAT_CMP_CONST, "strict comparison of f32 or f64 constant")
                    } else {
                        (FLOAT_CMP, "strict comparison of f32 or f64")
                    };
                    span_lint_and_then(cx, lint, expr.span, msg, |db| {
                        let lhs = Sugg::hir(cx, left, "..");
                        let rhs = Sugg::hir(cx, right, "..");

                        db.span_suggestion(
                            expr.span,
                            "consider comparing them within some error",
                            format!(
                                "({}).abs() {} error",
                                lhs - rhs,
                                if op == BinOpKind::Eq { '<' } else { '>' }
                            ),
                            Applicability::MachineApplicable, // snippet
                        );
                        db.span_note(expr.span, "std::f32::EPSILON and std::f64::EPSILON are available.");
                    });
                } else if op == BinOpKind::Rem && is_integer_literal(right, 1) {
                    span_lint(cx, MODULO_ONE, expr.span, "any number modulo 1 will be 0");
                }
            },
            _ => {},
        }
        if in_attributes_expansion(expr) {
            // Don't lint things expanded by #[derive(...)], etc
            return;
        }
        let binding = match expr.node {
            ExprKind::Path(ref qpath) => {
                let binding = last_path_segment(qpath).ident.as_str();
                if binding.starts_with('_') &&
                    !binding.starts_with("__") &&
                    binding != "_result" && // FIXME: #944
                    is_used(cx, expr) &&
                    // don't lint if the declaration is in a macro
                    non_macro_local(cx, cx.tables.qpath_res(qpath, expr.hir_id))
                {
                    Some(binding)
                } else {
                    None
                }
            },
            ExprKind::Field(_, ident) => {
                let name = ident.as_str();
                if name.starts_with('_') && !name.starts_with("__") {
                    Some(name)
                } else {
                    None
                }
            },
            _ => None,
        };
        if let Some(binding) = binding {
            span_lint(
                cx,
                USED_UNDERSCORE_BINDING,
                expr.span,
                &format!(
                    "used binding `{}` which is prefixed with an underscore. A leading \
                     underscore signals that a binding will not be used.",
                    binding
                ),
            );
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if let PatKind::Binding(.., ident, Some(ref right)) = pat.node {
            if let PatKind::Wild = right.node {
                span_lint(
                    cx,
                    REDUNDANT_PATTERN,
                    pat.span,
                    &format!(
                        "the `{} @ _` pattern can be written as just `{}`",
                        ident.name, ident.name
                    ),
                );
            }
        }
    }
}

fn check_nan(cx: &LateContext<'_, '_>, path: &Path, expr: &Expr) {
    if !in_constant(cx, expr.hir_id) {
        if let Some(seg) = path.segments.last() {
            if seg.ident.name == sym!(NAN) {
                span_lint(
                    cx,
                    CMP_NAN,
                    expr.span,
                    "doomed comparison with NAN, use `std::{f32,f64}::is_nan()` instead",
                );
            }
        }
    }
}

fn is_named_constant<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> bool {
    if let Some((_, res)) = constant(cx, cx.tables, expr) {
        res
    } else {
        false
    }
}

fn is_allowed<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> bool {
    match constant(cx, cx.tables, expr) {
        Some((Constant::F32(f), _)) => f == 0.0 || f.is_infinite(),
        Some((Constant::F64(f), _)) => f == 0.0 || f.is_infinite(),
        _ => false,
    }
}

fn is_float(cx: &LateContext<'_, '_>, expr: &Expr) -> bool {
    matches!(walk_ptrs_ty(cx.tables.expr_ty(expr)).sty, ty::Float(_))
}

fn check_to_owned(cx: &LateContext<'_, '_>, expr: &Expr, other: &Expr) {
    let (arg_ty, snip) = match expr.node {
        ExprKind::MethodCall(.., ref args) if args.len() == 1 => {
            if match_trait_method(cx, expr, &paths::TO_STRING) || match_trait_method(cx, expr, &paths::TO_OWNED) {
                (cx.tables.expr_ty_adjusted(&args[0]), snippet(cx, args[0].span, ".."))
            } else {
                return;
            }
        },
        ExprKind::Call(ref path, ref v) if v.len() == 1 => {
            if let ExprKind::Path(ref path) = path.node {
                if match_qpath(path, &["String", "from_str"]) || match_qpath(path, &["String", "from"]) {
                    (cx.tables.expr_ty_adjusted(&v[0]), snippet(cx, v[0].span, ".."))
                } else {
                    return;
                }
            } else {
                return;
            }
        },
        _ => return,
    };

    let other_ty = cx.tables.expr_ty_adjusted(other);
    let partial_eq_trait_id = match cx.tcx.lang_items().eq_trait() {
        Some(id) => id,
        None => return,
    };

    let deref_arg_impl_partial_eq_other = arg_ty.builtin_deref(true).map_or(false, |tam| {
        implements_trait(cx, tam.ty, partial_eq_trait_id, &[other_ty.into()])
    });
    let arg_impl_partial_eq_deref_other = other_ty.builtin_deref(true).map_or(false, |tam| {
        implements_trait(cx, arg_ty, partial_eq_trait_id, &[tam.ty.into()])
    });
    let arg_impl_partial_eq_other = implements_trait(cx, arg_ty, partial_eq_trait_id, &[other_ty.into()]);

    if !deref_arg_impl_partial_eq_other && !arg_impl_partial_eq_deref_other && !arg_impl_partial_eq_other {
        return;
    }

    let other_gets_derefed = match other.node {
        ExprKind::Unary(UnDeref, _) => true,
        _ => false,
    };

    let lint_span = if other_gets_derefed {
        expr.span.to(other.span)
    } else {
        expr.span
    };

    span_lint_and_then(
        cx,
        CMP_OWNED,
        lint_span,
        "this creates an owned instance just for comparison",
        |db| {
            // This also catches `PartialEq` implementations that call `to_owned`.
            if other_gets_derefed {
                db.span_label(lint_span, "try implementing the comparison without allocating");
                return;
            }

            let try_hint = if deref_arg_impl_partial_eq_other {
                // suggest deref on the left
                format!("*{}", snip)
            } else {
                // suggest dropping the to_owned on the left
                snip.to_string()
            };

            db.span_suggestion(
                lint_span,
                "try",
                try_hint,
                Applicability::MachineApplicable, // snippet
            );
        },
    );
}

/// Heuristic to see if an expression is used. Should be compatible with
/// `unused_variables`'s idea
/// of what it means for an expression to be "used".
fn is_used(cx: &LateContext<'_, '_>, expr: &Expr) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr) {
        match parent.node {
            ExprKind::Assign(_, ref rhs) | ExprKind::AssignOp(_, _, ref rhs) => SpanlessEq::new(cx).eq_expr(rhs, expr),
            _ => is_used(cx, parent),
        }
    } else {
        true
    }
}

/// Tests whether an expression is in a macro expansion (e.g., something
/// generated by `#[derive(...)]` or the like).
fn in_attributes_expansion(expr: &Expr) -> bool {
    expr.span
        .ctxt()
        .outer_expn_info()
        .map_or(false, |info| matches!(info.format, ExpnFormat::MacroAttribute(_)))
}

/// Tests whether `res` is a variable defined outside a macro.
fn non_macro_local(cx: &LateContext<'_, '_>, res: def::Res) -> bool {
    if let def::Res::Local(id) = res {
        !in_macro_or_desugar(cx.tcx.hir().span(id))
    } else {
        false
    }
}

fn check_cast(cx: &LateContext<'_, '_>, span: Span, e: &Expr, ty: &Ty) {
    if_chain! {
        if let TyKind::Ptr(MutTy { mutbl, .. }) = ty.node;
        if let ExprKind::Lit(ref lit) = e.node;
        if let LitKind::Int(value, ..) = lit.node;
        if value == 0;
        if !in_constant(cx, e.hir_id);
        then {
            let msg = match mutbl {
                Mutability::MutMutable => "`0 as *mut _` detected. Consider using `ptr::null_mut()`",
                Mutability::MutImmutable => "`0 as *const _` detected. Consider using `ptr::null()`",
            };
            span_lint(cx, ZERO_PTR, span, msg);
        }
    }
}
