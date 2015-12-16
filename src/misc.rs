use rustc::lint::*;
use syntax::ptr::P;
use rustc_front::hir::*;
use reexport::*;
use rustc_front::util::{is_comparison_binop, binop_to_string};
use syntax::codemap::{Span, Spanned};
use rustc_front::intravisit::FnKind;
use rustc::middle::ty;
use rustc::middle::const_eval::ConstVal::Float;
use rustc::middle::const_eval::eval_const_expr_partial;
use rustc::middle::const_eval::EvalHint::ExprTypeChecked;

use utils::{get_item_name, match_path, snippet, span_lint, walk_ptrs_ty, is_integer_literal};
use utils::span_help_and_lint;

/// **What it does:** This lint checks for function arguments and let bindings denoted as `ref`. It is `Warn` by default.
///
/// **Why is this bad?** The `ref` declaration makes the function take an owned value, but turns the argument into a reference (which means that the value is destroyed when exiting the function). This adds not much value: either take a reference type, or take an owned value and create references in the body.
///
/// For let bindings, `let x = &foo;` is preferred over `let ref x = foo`. The type of `x` is more obvious with the former.
///
/// **Known problems:** If the argument is dereferenced within the function, removing the `ref` will lead to errors. This can be fixed by removing the dereferences, e.g. changing `*x` to `x` within the function.
///
/// **Example:** `fn foo(ref x: u8) -> bool { .. }`
declare_lint!(pub TOPLEVEL_REF_ARG, Warn,
              "An entire binding was declared as `ref`, in a function argument (`fn foo(ref x: Bar)`), \
               or a `let` statement (`let ref x = foo()`). In such cases, it is preferred to take \
               references with `&`.");

#[allow(missing_copy_implementations)]
pub struct TopLevelRefPass;

impl LintPass for TopLevelRefPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOPLEVEL_REF_ARG)
    }
}

impl LateLintPass for TopLevelRefPass {
    fn check_fn(&mut self, cx: &LateContext, k: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        if let FnKind::Closure = k {
            // Does not apply to closures
            return
        }
        for ref arg in &decl.inputs {
            if let PatIdent(BindByRef(_), _, _) = arg.pat.node {
                span_lint(cx,
                    TOPLEVEL_REF_ARG,
                    arg.pat.span,
                    "`ref` directly on a function argument is ignored. Consider using a reference type instead."
                );
            }
        }
    }
    fn check_stmt(&mut self, cx: &LateContext, s: &Stmt) {
        if_let_chain! {
            [
            let StmtDecl(ref d, _) = s.node,
            let DeclLocal(ref l) = d.node,
            let PatIdent(BindByRef(_), i, None) = l.pat.node,
            let Some(ref init) = l.init
            ], {
                let tyopt = if let Some(ref ty) = l.ty {
                    format!(": {:?} ", ty)
                } else {
                    "".to_owned()
                };
                span_help_and_lint(cx,
                    TOPLEVEL_REF_ARG,
                    l.pat.span,
                    "`ref` on an entire `let` pattern is discouraged, take a reference with & instead",
                    &format!("try `let {} {}= &{};`", snippet(cx, i.span, "_"),
                             tyopt, snippet(cx, init.span, "_"))
                );
            }
        };
    }
}

/// **What it does:** This lint checks for comparisons to NAN. It is `Deny` by default.
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
            if is_comparison_binop(cmp.node) {
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
    path.segments.last().map(|seg| if seg.identifier.name.as_str() == "NAN" {
        span_lint(cx, CMP_NAN, span,
            "doomed comparison with NAN, use `std::{f32,f64}::is_nan()` instead");
    });
}

/// **What it does:** This lint checks for (in-)equality comparisons on floating-point values (apart from zero), except in functions called `*eq*` (which probably implement equality for a type involving floats). It is `Warn` by default.
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
                if is_allowed(cx, left) || is_allowed(cx, right) { return; }
                if let Some(name) = get_item_name(cx, expr) {
                    let name = name.as_str();
                    if name == "eq" || name == "ne" || name == "is_nan" ||
                            name.starts_with("eq_") ||
                            name.ends_with("_eq") {
                        return;
                    }
                }
                span_lint(cx, FLOAT_CMP, expr.span, &format!(
                    "{}-comparison of f32 or f64 detected. Consider changing this to \
                     `abs({} - {}) < epsilon` for some suitable value of epsilon",
                    binop_to_string(op), snippet(cx, left.span, ".."),
                    snippet(cx, right.span, "..")));
            }
        }
    }
}

fn is_allowed(cx: &LateContext, expr: &Expr) -> bool {
    let res = eval_const_expr_partial(cx.tcx, expr, ExprTypeChecked, None);
    if let Ok(Float(val)) = res {
        val == 0.0 || val == ::std::f64::INFINITY || val == ::std::f64::NEG_INFINITY
    } else { false }
}

fn is_float(cx: &LateContext, expr: &Expr) -> bool {
    if let ty::TyFloat(_) = walk_ptrs_ty(cx.tcx.expr_ty(expr)).sty {
        true
    } else {
        false
    }
}

/// **What it does:** This lint checks for conversions to owned values just for the sake of a comparison. It is `Warn` by default.
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
            if is_comparison_binop(cmp.node) {
                check_to_owned(cx, left, right.span, true, cmp.span);
                check_to_owned(cx, right, left.span, false, cmp.span)
            }
        }
    }
}

fn check_to_owned(cx: &LateContext, expr: &Expr, other_span: Span, left: bool, op: Span) {
    let snip = match expr.node {
        ExprMethodCall(Spanned{node: ref name, ..}, _, ref args) if args.len() == 1 => {
            if name.as_str() == "to_string" ||
                name.as_str() == "to_owned" && is_str_arg(cx, args) {
                    snippet(cx, args[0].span, "..")
                } else {
                    return
                }
        }
        ExprCall(ref path, ref v) if v.len() == 1 => {
            if let ExprPath(None, ref path) = path.node {
                if match_path(path, &["String", "from_str"]) ||
                    match_path(path, &["String", "from"]) {
                            snippet(cx, v[0].span, "..")
                    } else {
                        return
                    }
            } else {
                return
            }
        }
        _ => return
    };
    if left {
        span_lint(cx, CMP_OWNED, expr.span, &format!(
        "this creates an owned instance just for comparison. Consider using \
        `{} {} {}` to compare without allocation", snip,
        snippet(cx, op, "=="), snippet(cx, other_span, "..")));
    } else {
        span_lint(cx, CMP_OWNED, expr.span, &format!(
        "this creates an owned instance just for comparison. Consider using \
        `{} {} {}` to compare without allocation",
        snippet(cx, other_span, ".."), snippet(cx, op, "=="),  snip));
    }

}

fn is_str_arg(cx: &LateContext, args: &[P<Expr>]) -> bool {
    args.len() == 1 && if let ty::TyStr =
        walk_ptrs_ty(cx.tcx.expr_ty(&args[0])).sty { true } else { false }
}

/// **What it does:** This lint checks for getting the remainder of a division by one. It is `Warn` by default.
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
            if let Spanned {node: BinOp_::BiRem, ..} = *cmp {
                if is_integer_literal(right, 1) {
                    cx.span_lint(MODULO_ONE, expr.span, "any number modulo 1 will be 0");
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
        if let PatIdent(_, ref ident, Some(ref right)) = pat.node {
            if right.node == PatWild {
                cx.span_lint(REDUNDANT_PATTERN, pat.span, &format!(
                    "the `{} @ _` pattern can be written as just `{}`",
                    ident.node.name, ident.node.name));
            }
        }
    }
}


/// **What it does:** This lint checks for the use of bindings with a single leading underscore
///
/// **Why is this bad?** A single leading underscore is usually used to indicate that a binding
/// will not be used. Using such a binding breaks this expectation.
///
/// **Known problems:** This lint's idea of a "used" variable is not quite the same as in the
/// built-in `unused_variables` lint. For example, in the following code
/// ```
/// fn foo(y: u32) -> u32) {
///     let _x = 1;
///     _x +=1;
///     y
/// }
/// ```
/// _x will trigger both the `unused_variables` lint and the `used_underscore_binding` lint.
///
/// **Example**:
/// ```
/// let _x = 0;
/// let y = _x + 1; // Here we are using `_x`, even though it has a leading underscore.
///                 // We should rename `_x` to `x`
/// ```
declare_lint!(pub USED_UNDERSCORE_BINDING, Warn,
              "using a binding which is prefixed with an underscore");

#[derive(Copy, Clone)]
pub struct UsedUnderscoreBinding;

impl LintPass for UsedUnderscoreBinding {
    fn get_lints(&self) -> LintArray {
        lint_array!(USED_UNDERSCORE_BINDING)
    }
}

impl LateLintPass for UsedUnderscoreBinding {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        let needs_lint = match expr.node {
            ExprPath(_, ref path) => {
                let ident = path.segments.last()
                                .expect("path should always have at least one segment")
                                .identifier;
                ident.name.as_str().chars().next() == Some('_') //starts with '_'
                && ident.name.as_str().chars().skip(1).next() != Some('_') //doesn't start with "__"
                && ident.name != ident.unhygienic_name //not in macro
                && cx.tcx.def_map.borrow().contains_key(&expr.id) //local variable
            },
            ExprField(_, spanned) => {
                let name = spanned.node.as_str();
                name.chars().next() == Some('_')
                && name.chars().skip(1).next() != Some('_')
            },
            _ => false
        };
        if needs_lint {
            cx.span_lint(USED_UNDERSCORE_BINDING, expr.span,
                         "used binding which is prefixed with an underscore. A leading underscore\
                          signals that a binding will not be used.");
        }
    }
}
