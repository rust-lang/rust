use reexport::*;
use rustc::hir::*;
use rustc::hir::intravisit::FnKind;
use rustc::lint::*;
use rustc::middle::const_val::ConstVal;
use rustc::ty;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use rustc_const_eval::eval_const_expr_partial;
use syntax::codemap::{Span, Spanned, ExpnFormat};
use syntax::ptr::P;
use utils::{
    get_item_name, get_parent_expr, implements_trait, is_integer_literal, match_path, snippet,
    span_lint, span_lint_and_then, walk_ptrs_ty
};

/// **What it does:** This lint checks for function arguments and let bindings denoted as `ref`.
///
/// **Why is this bad?** The `ref` declaration makes the function take an owned value, but turns the argument into a reference (which means that the value is destroyed when exiting the function). This adds not much value: either take a reference type, or take an owned value and create references in the body.
///
/// For let bindings, `let x = &foo;` is preferred over `let ref x = foo`. The type of `x` is more obvious with the former.
///
/// **Known problems:** If the argument is dereferenced within the function, removing the `ref` will lead to errors. This can be fixed by removing the dereferences, e.g. changing `*x` to `x` within the function.
///
/// **Example:** `fn foo(ref x: u8) -> bool { .. }`
declare_lint! {
    pub TOPLEVEL_REF_ARG, Warn,
    "An entire binding was declared as `ref`, in a function argument (`fn foo(ref x: Bar)`), \
     or a `let` statement (`let ref x = foo()`). In such cases, it is preferred to take \
     references with `&`."
}

#[allow(missing_copy_implementations)]
pub struct TopLevelRefPass;

impl LintPass for TopLevelRefPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOPLEVEL_REF_ARG)
    }
}

impl LateLintPass for TopLevelRefPass {
    fn check_fn(&mut self, cx: &LateContext, k: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        if let FnKind::Closure(_) = k {
            // Does not apply to closures
            return;
        }
        for ref arg in &decl.inputs {
            if let PatKind::Binding(BindByRef(_), _, _) = arg.pat.node {
                span_lint(cx,
                          TOPLEVEL_REF_ARG,
                          arg.pat.span,
                          "`ref` directly on a function argument is ignored. Consider using a reference type instead.");
            }
        }
    }
    fn check_stmt(&mut self, cx: &LateContext, s: &Stmt) {
        if_let_chain! {
            [
            let StmtDecl(ref d, _) = s.node,
            let DeclLocal(ref l) = d.node,
            let PatKind::Binding(BindByRef(_), i, None) = l.pat.node,
            let Some(ref init) = l.init
            ], {
                let tyopt = if let Some(ref ty) = l.ty {
                    format!(": {}", snippet(cx, ty.span, "_"))
                } else {
                    "".to_owned()
                };
                span_lint_and_then(cx,
                    TOPLEVEL_REF_ARG,
                    l.pat.span,
                    "`ref` on an entire `let` pattern is discouraged, take a reference with & instead",
                    |db| {
                        db.span_suggestion(s.span,
                                           "try",
                                           format!("let {}{} = &{};",
                                                   snippet(cx, i.span, "_"),
                                                   tyopt,
                                                   snippet(cx, init.span, "_")));
                    }
                );
            }
        };
    }
}

/// **What it does:** This lint checks for comparisons to NAN.
///
/// **Why is this bad?** NAN does not compare meaningfully to anything – not even itself – so those comparisons are simply wrong.
///
/// **Known problems:** None
///
/// **Example:** `x == NAN`
declare_lint!(pub CMP_NAN, Deny,
              "comparisons to NAN (which will always return false, which is probably not intended)");

#[derive(Copy,Clone)]
pub struct CmpNan;

impl LintPass for CmpNan {
    fn get_lints(&self) -> LintArray {
        lint_array!(CMP_NAN)
    }
}

impl LateLintPass for CmpNan {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprBinary(ref cmp, ref left, ref right) = expr.node {
            if cmp.node.is_comparison() {
                if let ExprPath(_, ref path) = left.node {
                    check_nan(cx, path, expr.span);
                }
                if let ExprPath(_, ref path) = right.node {
                    check_nan(cx, path, expr.span);
                }
            }
        }
    }
}

fn check_nan(cx: &LateContext, path: &Path, span: Span) {
    path.segments.last().map(|seg| {
        if seg.name.as_str() == "NAN" {
            span_lint(cx,
                      CMP_NAN,
                      span,
                      "doomed comparison with NAN, use `std::{f32,f64}::is_nan()` instead");
        }
    });
}

/// **What it does:** This lint checks for (in-)equality comparisons on floating-point values (apart from zero), except in functions called `*eq*` (which probably implement equality for a type involving floats).
///
/// **Why is this bad?** Floating point calculations are usually imprecise, so asking if two values are *exactly* equal is asking for trouble. For a good guide on what to do, see [the floating point guide](http://www.floating-point-gui.de/errors/comparison).
///
/// **Known problems:** None
///
/// **Example:** `y == 1.23f64`
declare_lint!(pub FLOAT_CMP, Warn,
              "using `==` or `!=` on float values (as floating-point operations \
               usually involve rounding errors, it is always better to check for approximate \
               equality within small bounds)");

#[derive(Copy,Clone)]
pub struct FloatCmp;

impl LintPass for FloatCmp {
    fn get_lints(&self) -> LintArray {
        lint_array!(FLOAT_CMP)
    }
}

impl LateLintPass for FloatCmp {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprBinary(ref cmp, ref left, ref right) = expr.node {
            let op = cmp.node;
            if (op == BiEq || op == BiNe) && (is_float(cx, left) || is_float(cx, right)) {
                if is_allowed(cx, left) || is_allowed(cx, right) {
                    return;
                }
                if let Some(name) = get_item_name(cx, expr) {
                    let name = name.as_str();
                    if name == "eq" || name == "ne" || name == "is_nan" || name.starts_with("eq_") ||
                       name.ends_with("_eq") {
                        return;
                    }
                }
                span_lint(cx,
                          FLOAT_CMP,
                          expr.span,
                          &format!("{}-comparison of f32 or f64 detected. Consider changing this to `({} - {}).abs() < \
                                    epsilon` for some suitable value of epsilon. \
                                    std::f32::EPSILON and std::f64::EPSILON are available.",
                                   op.as_str(),
                                   snippet(cx, left.span, ".."),
                                   snippet(cx, right.span, "..")));
            }
        }
    }
}

fn is_allowed(cx: &LateContext, expr: &Expr) -> bool {
    let res = eval_const_expr_partial(cx.tcx, expr, ExprTypeChecked, None);
    if let Ok(ConstVal::Float(val)) = res {
        val == 0.0 || val == ::std::f64::INFINITY || val == ::std::f64::NEG_INFINITY
    } else {
        false
    }
}

fn is_float(cx: &LateContext, expr: &Expr) -> bool {
    if let ty::TyFloat(_) = walk_ptrs_ty(cx.tcx.expr_ty(expr)).sty {
        true
    } else {
        false
    }
}

/// **What it does:** This lint checks for conversions to owned values just for the sake of a comparison.
///
/// **Why is this bad?** The comparison can operate on a reference, so creating an owned value effectively throws it away directly afterwards, which is needlessly consuming code and heap space.
///
/// **Known problems:** None
///
/// **Example:** `x.to_owned() == y`
declare_lint!(pub CMP_OWNED, Warn,
              "creating owned instances for comparing with others, e.g. `x == \"foo\".to_string()`");

#[derive(Copy,Clone)]
pub struct CmpOwned;

impl LintPass for CmpOwned {
    fn get_lints(&self) -> LintArray {
        lint_array!(CMP_OWNED)
    }
}

impl LateLintPass for CmpOwned {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprBinary(ref cmp, ref left, ref right) = expr.node {
            if cmp.node.is_comparison() {
                check_to_owned(cx, left, right, true, cmp.span);
                check_to_owned(cx, right, left, false, cmp.span)
            }
        }
    }
}

fn check_to_owned(cx: &LateContext, expr: &Expr, other: &Expr, left: bool, op: Span) {
    let (arg_ty, snip) = match expr.node {
        ExprMethodCall(Spanned { node: ref name, .. }, _, ref args) if args.len() == 1 => {
            if name.as_str() == "to_string" || name.as_str() == "to_owned" && is_str_arg(cx, args) {
                (cx.tcx.expr_ty(&args[0]), snippet(cx, args[0].span, ".."))
            } else {
                return;
            }
        }
        ExprCall(ref path, ref v) if v.len() == 1 => {
            if let ExprPath(None, ref path) = path.node {
                if match_path(path, &["String", "from_str"]) || match_path(path, &["String", "from"]) {
                    (cx.tcx.expr_ty(&v[0]), snippet(cx, v[0].span, ".."))
                } else {
                    return;
                }
            } else {
                return;
            }
        }
        _ => return,
    };

    let other_ty = cx.tcx.expr_ty(other);
    let partial_eq_trait_id = match cx.tcx.lang_items.eq_trait() {
        Some(id) => id,
        None => return,
    };

    if !implements_trait(cx, arg_ty, partial_eq_trait_id, vec![other_ty]) {
        return;
    }

    if left {
        span_lint(cx,
                  CMP_OWNED,
                  expr.span,
                  &format!("this creates an owned instance just for comparison. Consider using `{} {} {}` to \
                            compare without allocation",
                           snip,
                           snippet(cx, op, "=="),
                           snippet(cx, other.span, "..")));
    } else {
        span_lint(cx,
                  CMP_OWNED,
                  expr.span,
                  &format!("this creates an owned instance just for comparison. Consider using `{} {} {}` to \
                            compare without allocation",
                           snippet(cx, other.span, ".."),
                           snippet(cx, op, "=="),
                           snip));
    }

}

fn is_str_arg(cx: &LateContext, args: &[P<Expr>]) -> bool {
    args.len() == 1 &&
    if let ty::TyStr = walk_ptrs_ty(cx.tcx.expr_ty(&args[0])).sty {
        true
    } else {
        false
    }
}

/// **What it does:** This lint checks for getting the remainder of a division by one.
///
/// **Why is this bad?** The result can only ever be zero. No one will write such code deliberately, unless trying to win an Underhanded Rust Contest. Even for that contest, it's probably a bad idea. Use something more underhanded.
///
/// **Known problems:** None
///
/// **Example:** `x % 1`
declare_lint!(pub MODULO_ONE, Warn, "taking a number modulo 1, which always returns 0");

#[derive(Copy,Clone)]
pub struct ModuloOne;

impl LintPass for ModuloOne {
    fn get_lints(&self) -> LintArray {
        lint_array!(MODULO_ONE)
    }
}

impl LateLintPass for ModuloOne {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprBinary(ref cmp, _, ref right) = expr.node {
            if let Spanned { node: BinOp_::BiRem, .. } = *cmp {
                if is_integer_literal(right, 1) {
                    span_lint(cx, MODULO_ONE, expr.span, "any number modulo 1 will be 0");
                }
            }
        }
    }
}

/// **What it does:** This lint checks for patterns in the form `name @ _`.
///
/// **Why is this bad?** It's almost always more readable to just use direct bindings.
///
/// **Known problems:** None
///
/// **Example**:
/// ```
/// match v {
///     Some(x) => (),
///     y @ _   => (), // easier written as `y`,
/// }
/// ```
declare_lint!(pub REDUNDANT_PATTERN, Warn, "using `name @ _` in a pattern");

#[derive(Copy,Clone)]
pub struct PatternPass;

impl LintPass for PatternPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_PATTERN)
    }
}

impl LateLintPass for PatternPass {
    fn check_pat(&mut self, cx: &LateContext, pat: &Pat) {
        if let PatKind::Binding(_, ref ident, Some(ref right)) = pat.node {
            if right.node == PatKind::Wild {
                span_lint(cx,
                          REDUNDANT_PATTERN,
                          pat.span,
                          &format!("the `{} @ _` pattern can be written as just `{}`",
                                   ident.node,
                                   ident.node));
            }
        }
    }
}


/// **What it does:** This lint checks for the use of bindings with a single leading underscore
///
/// **Why is this bad?** A single leading underscore is usually used to indicate that a binding
/// will not be used. Using such a binding breaks this expectation.
///
/// **Known problems:** The lint does not work properly with desugaring and macro, it has been
/// allowed in the mean time.
///
/// **Example**:
/// ```
/// let _x = 0;
/// let y = _x + 1; // Here we are using `_x`, even though it has a leading underscore.
///                 // We should rename `_x` to `x`
/// ```
declare_lint!(pub USED_UNDERSCORE_BINDING, Allow,
              "using a binding which is prefixed with an underscore");

#[derive(Copy, Clone)]
pub struct UsedUnderscoreBinding;

impl LintPass for UsedUnderscoreBinding {
    fn get_lints(&self) -> LintArray {
        lint_array!(USED_UNDERSCORE_BINDING)
    }
}

impl LateLintPass for UsedUnderscoreBinding {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if in_attributes_expansion(cx, expr) {
            // Don't lint things expanded by #[derive(...)], etc
            return;
        }
        let binding = match expr.node {
            ExprPath(_, ref path) => {
                let segment = path.segments
                                .last()
                                .expect("path should always have at least one segment")
                                .name;
                if segment.as_str().starts_with('_') &&
                   !segment.as_str().starts_with("__") &&
                   segment != segment.unhygienize() && // not in bang macro
                   is_used(cx, expr) {
                    Some(segment.as_str())
                } else {
                    None
                }
            }
            ExprField(_, spanned) => {
                let name = spanned.node.as_str();
                if name.starts_with('_') && !name.starts_with("__") {
                    Some(name)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(binding) = binding {
            if binding != "_result" { // FIXME: #944
                span_lint(cx,
                          USED_UNDERSCORE_BINDING,
                          expr.span,
                          &format!("used binding `{}` which is prefixed with an underscore. A leading \
                                    underscore signals that a binding will not be used.", binding));
            }
        }
    }
}

/// Heuristic to see if an expression is used. Should be compatible with `unused_variables`'s idea
/// of what it means for an expression to be "used".
fn is_used(cx: &LateContext, expr: &Expr) -> bool {
    if let Some(ref parent) = get_parent_expr(cx, expr) {
        match parent.node {
            ExprAssign(_, ref rhs) |
            ExprAssignOp(_, _, ref rhs) => **rhs == *expr,
            _ => is_used(cx, parent),
        }
    } else {
        true
    }
}

/// Test whether an expression is in a macro expansion (e.g. something generated by
/// `#[derive(...)`] or the like).
fn in_attributes_expansion(cx: &LateContext, expr: &Expr) -> bool {
    cx.sess().codemap().with_expn_info(expr.span.expn_id, |info_opt| {
        info_opt.map_or(false, |info| {
            match info.callee.format {
                ExpnFormat::MacroAttribute(_) => true,
                _ => false,
            }
        })
    })
}
