use reexport::*;
use rustc::hir::*;
use rustc::hir::intravisit::FnKind;
use rustc::lint::*;
use rustc::ty;
use syntax::codemap::{ExpnFormat, Span};
use utils::{get_item_name, get_parent_expr, implements_trait, in_constant, in_macro, is_integer_literal,
            iter_input_pats, last_path_segment, match_qpath, match_trait_method, paths, snippet, span_lint,
            span_lint_and_then, walk_ptrs_ty};
use utils::sugg::Sugg;
use syntax::ast::{LitKind, CRATE_NODE_ID};
use consts::{constant, Constant};

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
/// dereferences, e.g. changing `*x` to `x` within the function.
///
/// **Example:**
/// ```rust
/// fn foo(ref x: u8) -> bool { .. }
/// ```
declare_clippy_lint! {
    pub TOPLEVEL_REF_ARG,
    style,
    "an entire binding declared as `ref`, in a function argument or a `let` statement"
}

/// **What it does:** Checks for comparisons to NaN.
///
/// **Why is this bad?** NaN does not compare meaningfully to anything – not
/// even itself – so those comparisons are simply wrong.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x == NAN
/// ```
declare_clippy_lint! {
    pub CMP_NAN,
    correctness,
    "comparisons to NAN, which will always return false, probably not intended"
}

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
/// y == 1.23f64
/// y != x  // where both are floats
/// ```
declare_clippy_lint! {
    pub FLOAT_CMP,
    correctness,
    "using `==` or `!=` on float values instead of comparing difference with an epsilon"
}

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
declare_clippy_lint! {
    pub CMP_OWNED,
    perf,
    "creating owned instances for comparing with others, e.g. `x == \"foo\".to_string()`"
}

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
/// x % 1
/// ```
declare_clippy_lint! {
    pub MODULO_ONE,
    correctness,
    "taking a number modulo 1, which always returns 0"
}

/// **What it does:** Checks for patterns in the form `name @ _`.
///
/// **Why is this bad?** It's almost always more readable to just use direct
/// bindings.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// match v {
///     Some(x) => (),
///     y @ _   => (), // easier written as `y`,
/// }
/// ```
declare_clippy_lint! {
    pub REDUNDANT_PATTERN,
    style,
    "using `name @ _` in a pattern"
}

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
declare_clippy_lint! {
    pub USED_UNDERSCORE_BINDING,
    pedantic,
    "using a binding which is prefixed with an underscore"
}

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
/// f() && g();  // We should write `if f() { g(); }`.
/// ```
declare_clippy_lint! {
    pub SHORT_CIRCUIT_STATEMENT,
    complexity,
    "using a short circuit boolean condition as a statement"
}

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
/// 0 as *const u32
/// ```
declare_clippy_lint! {
    pub ZERO_PTR,
    style,
    "using 0 as *{const, mut} T"
}

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
/// const ONE == 1.00f64
/// x == ONE  // where both are floats
/// ```
declare_clippy_lint! {
    pub FLOAT_CMP_CONST,
    restriction,
    "using `==` or `!=` on float constants instead of comparing difference with an epsilon"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(
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
        )
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        k: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        _: NodeId,
    ) {
        if let FnKind::Closure(_) = k {
            // Does not apply to closures
            return;
        }
        for arg in iter_input_pats(decl, body) {
            match arg.pat.node {
                PatKind::Binding(BindingAnnotation::Ref, _, _, _) |
                PatKind::Binding(BindingAnnotation::RefMut, _, _, _) => {
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
            if let StmtDecl(ref d, _) = s.node;
            if let DeclLocal(ref l) = d.node;
            if let PatKind::Binding(an, _, i, None) = l.pat.node;
            if let Some(ref init) = l.init;
            then {
                if an == BindingAnnotation::Ref || an == BindingAnnotation::RefMut {
                    let init = Sugg::hir(cx, init, "..");
                    let (mutopt,initref) = if an == BindingAnnotation::RefMut {
                        ("mut ", init.mut_addr())
                    } else {
                        ("", init.addr())
                    };
                    let tyopt = if let Some(ref ty) = l.ty {
                        format!(": &{mutopt}{ty}", mutopt=mutopt, ty=snippet(cx, ty.span, "_"))
                    } else {
                        "".to_owned()
                    };
                    span_lint_and_then(cx,
                        TOPLEVEL_REF_ARG,
                        l.pat.span,
                        "`ref` on an entire `let` pattern is discouraged, take a reference with `&` instead",
                        |db| {
                            db.span_suggestion(s.span,
                                               "try",
                                               format!("let {name}{tyopt} = {initref};",
                                                       name=snippet(cx, i.span, "_"),
                                                       tyopt=tyopt,
                                                       initref=initref));
                        }
                    );
                }
            }
        };
        if_chain! {
            if let StmtSemi(ref expr, _) = s.node;
            if let Expr_::ExprBinary(ref binop, ref a, ref b) = expr.node;
            if binop.node == BiAnd || binop.node == BiOr;
            if let Some(sugg) = Sugg::hir_opt(cx, a);
            then {
                span_lint_and_then(cx,
                    SHORT_CIRCUIT_STATEMENT,
                    s.span,
                    "boolean short circuit operator in statement may be clearer using an explicit test",
                    |db| {
                        let sugg = if binop.node == BiOr { !sugg } else { sugg };
                        db.span_suggestion(s.span, "replace it with",
                                           format!("if {} {{ {}; }}", sugg, &snippet(cx, b.span, "..")));
                    });
            }
        };
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        match expr.node {
            ExprCast(ref e, ref ty) => {
                check_cast(cx, expr.span, e, ty);
                return;
            },
            ExprBinary(ref cmp, ref left, ref right) => {
                let op = cmp.node;
                if op.is_comparison() {
                    if let ExprPath(QPath::Resolved(_, ref path)) = left.node {
                        check_nan(cx, path, expr);
                    }
                    if let ExprPath(QPath::Resolved(_, ref path)) = right.node {
                        check_nan(cx, path, expr);
                    }
                    check_to_owned(cx, left, right);
                    check_to_owned(cx, right, left);
                }
                if (op == BiEq || op == BiNe) && (is_float(cx, left) || is_float(cx, right)) {
                    if is_allowed(cx, left) || is_allowed(cx, right) {
                        return;
                    }
                    if let Some(name) = get_item_name(cx, expr) {
                        let name = name.as_str();
                        if name == "eq" || name == "ne" || name == "is_nan" || name.starts_with("eq_")
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
                            format!("({}).abs() < error", lhs - rhs),
                        );
                        db.span_note(expr.span, "std::f32::EPSILON and std::f64::EPSILON are available.");
                    });
                } else if op == BiRem && is_integer_literal(right, 1) {
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
            ExprPath(ref qpath) => {
                let binding = last_path_segment(qpath).name.as_str();
                if binding.starts_with('_') &&
                    !binding.starts_with("__") &&
                    binding != "_result" && // FIXME: #944
                    is_used(cx, expr) &&
                    // don't lint if the declaration is in a macro
                    non_macro_local(cx, &cx.tables.qpath_def(qpath, expr.hir_id))
                {
                    Some(binding)
                } else {
                    None
                }
            },
            ExprField(_, spanned) => {
                let name = spanned.node.as_str();
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
        if let PatKind::Binding(_, _, ref ident, Some(ref right)) = pat.node {
            if right.node == PatKind::Wild {
                span_lint(
                    cx,
                    REDUNDANT_PATTERN,
                    pat.span,
                    &format!("the `{} @ _` pattern can be written as just `{}`", ident.node, ident.node),
                );
            }
        }
    }
}

fn check_nan(cx: &LateContext, path: &Path, expr: &Expr) {
    if !in_constant(cx, expr.id) {
        if let Some(seg) = path.segments.last() {
            path.segments.last().map(|seg| {
                if seg.name == "NAN" {
                    span_lint(
                        cx,
                        CMP_NAN,
                        expr.span,
                        "doomed comparison with NAN, use `std::{f32,f64}::is_nan()` instead",
                    );
                }
            });
        }
    }
}

fn is_named_constant<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> bool {
    if let Some((_, res)) = constant(cx, expr) {
        res
    } else {
       false
    }
}

fn is_allowed<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> bool {
    match constant(cx, expr) {
        Some((Constant::F32(f), _)) => f == 0.0 || f.is_infinite(),
        Some((Constant::F64(f), _)) => f == 0.0 || f.is_infinite(),
        _ => false,
    }
}

fn is_float(cx: &LateContext, expr: &Expr) -> bool {
    matches!(walk_ptrs_ty(cx.tables.expr_ty(expr)).sty, ty::TyFloat(_))
}

fn check_to_owned(cx: &LateContext, expr: &Expr, other: &Expr) {
    let (arg_ty, snip) = match expr.node {
        ExprMethodCall(.., ref args) if args.len() == 1 => {
            if match_trait_method(cx, expr, &paths::TO_STRING) || match_trait_method(cx, expr, &paths::TO_OWNED) {
                (cx.tables.expr_ty_adjusted(&args[0]), snippet(cx, args[0].span, ".."))
            } else {
                return;
            }
        },
        ExprCall(ref path, ref v) if v.len() == 1 => if let ExprPath(ref path) = path.node {
            if match_qpath(path, &["String", "from_str"]) || match_qpath(path, &["String", "from"]) {
                (cx.tables.expr_ty_adjusted(&v[0]), snippet(cx, v[0].span, ".."))
            } else {
                return;
            }
        } else {
            return;
        },
        _ => return,
    };

    let other_ty = cx.tables.expr_ty_adjusted(other);
    let partial_eq_trait_id = match cx.tcx.lang_items().eq_trait() {
        Some(id) => id,
        None => return,
    };

    // *arg impls PartialEq<other>
    if !arg_ty
        .builtin_deref(true)
        .map_or(false, |tam| implements_trait(cx, tam.ty, partial_eq_trait_id, &[other_ty]))
        // arg impls PartialEq<*other>
        && !other_ty
        .builtin_deref(true)
        .map_or(false, |tam| implements_trait(cx, arg_ty, partial_eq_trait_id, &[tam.ty]))
        // arg impls PartialEq<other>
        && !implements_trait(cx, arg_ty, partial_eq_trait_id, &[other_ty])
    {
        return;
    }

    span_lint_and_then(
        cx,
        CMP_OWNED,
        expr.span,
        "this creates an owned instance just for comparison",
        |db| {
            // this is as good as our recursion check can get, we can't prove that the
            // current function is
            // called by
            // PartialEq::eq, but we can at least ensure that this code is not part of it
            let parent_fn = cx.tcx.hir.get_parent(expr.id);
            let parent_impl = cx.tcx.hir.get_parent(parent_fn);
            if parent_impl != CRATE_NODE_ID {
                if let map::NodeItem(item) = cx.tcx.hir.get(parent_impl) {
                    if let ItemImpl(.., Some(ref trait_ref), _, _) = item.node {
                        if trait_ref.path.def.def_id() == partial_eq_trait_id {
                            // we are implementing PartialEq, don't suggest not doing `to_owned`, otherwise
                            // we go into
                            // recursion
                            db.span_label(expr.span, "try calling implementing the comparison without allocating");
                            return;
                        }
                    }
                }
            }
            db.span_suggestion(expr.span, "try", snip.to_string());
        },
    );
}

/// Heuristic to see if an expression is used. Should be compatible with
/// `unused_variables`'s idea
/// of what it means for an expression to be "used".
fn is_used(cx: &LateContext, expr: &Expr) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr) {
        match parent.node {
            ExprAssign(_, ref rhs) | ExprAssignOp(_, _, ref rhs) => **rhs == *expr,
            _ => is_used(cx, parent),
        }
    } else {
        true
    }
}

/// Test whether an expression is in a macro expansion (e.g. something
/// generated by
/// `#[derive(...)`] or the like).
fn in_attributes_expansion(expr: &Expr) -> bool {
    expr.span
        .ctxt()
        .outer()
        .expn_info()
        .map_or(false, |info| matches!(info.callee.format, ExpnFormat::MacroAttribute(_)))
}

/// Test whether `def` is a variable defined outside a macro.
fn non_macro_local(cx: &LateContext, def: &def::Def) -> bool {
    match *def {
        def::Def::Local(id) | def::Def::Upvar(id, _, _) => !in_macro(cx.tcx.hir.span(id)),
        _ => false,
    }
}

fn check_cast(cx: &LateContext, span: Span, e: &Expr, ty: &Ty) {
    if_chain! {
        if let TyPtr(MutTy { mutbl, .. }) = ty.node;
        if let ExprLit(ref lit) = e.node;
        if let LitKind::Int(value, ..) = lit.node;
        if value == 0;
        if !in_constant(cx, e.id);
        then {
            let msg = match mutbl {
                Mutability::MutMutable => "`0 as *mut _` detected. Consider using `ptr::null_mut()`",
                Mutability::MutImmutable => "`0 as *const _` detected. Consider using `ptr::null()`",
            };
            span_lint(cx, ZERO_PTR, span, msg);
        }
    }
}
