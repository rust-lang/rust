use rustc::lint::*;
use rustc::ty::{self, Ty};
use rustc::hir::*;
use utils::{match_def_path, paths, span_lint, span_lint_and_then, snippet, last_path_segment};
use utils::sugg;

/// **What it does:** Checks for transmutes that can't ever be correct on any
/// architecture.
///
/// **Why is this bad?** It's basically guaranteed to be undefined behaviour.
///
/// **Known problems:** When accessing C, users might want to store pointer
/// sized objects in `extradata` arguments to save an allocation.
///
/// **Example:**
/// ```rust
/// let ptr: *const T = core::intrinsics::transmute('x')`
/// ```
declare_lint! {
    pub WRONG_TRANSMUTE,
    Warn,
    "transmutes that are confusing at best, undefined behaviour at worst and always useless"
}

/// **What it does:** Checks for transmutes to the original type of the object
/// and transmutes that could be a cast.
///
/// **Why is this bad?** Readability. The code tricks people into thinking that
/// something complex is going on.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// core::intrinsics::transmute(t) // where the result type is the same as `t`'s
/// ```
declare_lint! {
    pub USELESS_TRANSMUTE,
    Warn,
    "transmutes that have the same to and from types or could be a cast/coercion"
}

/// **What it does:** Checks for transmutes between a type `T` and `*T`.
///
/// **Why is this bad?** It's easy to mistakenly transmute between a type and a
/// pointer to that type.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// core::intrinsics::transmute(t)` // where the result type is the same as `*t` or `&t`'s
/// ```
declare_lint! {
    pub CROSSPOINTER_TRANSMUTE,
    Warn,
    "transmutes that have to or from types that are a pointer to the other"
}

/// **What it does:** Checks for transmutes from a pointer to a reference.
///
/// **Why is this bad?** This can always be rewritten with `&` and `*`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let _: &T = std::mem::transmute(p); // where p: *const T
/// // can be written:
/// let _: &T = &*p;
/// ```
declare_lint! {
    pub TRANSMUTE_PTR_TO_REF,
    Warn,
    "transmutes from a pointer to a reference type"
}

pub struct Transmute;

impl LintPass for Transmute {
    fn get_lints(&self) -> LintArray {
        lint_array![CROSSPOINTER_TRANSMUTE, TRANSMUTE_PTR_TO_REF, USELESS_TRANSMUTE, WRONG_TRANSMUTE]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Transmute {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(ref qpath) = path_expr.node {
                let def_id = cx.tables.qpath_def(qpath, path_expr.id).def_id();

                if match_def_path(cx.tcx, def_id, &paths::TRANSMUTE) {
                    let from_ty = cx.tables.expr_ty(&args[0]);
                    let to_ty = cx.tables.expr_ty(e);

                    match (&from_ty.sty, &to_ty.sty) {
                        _ if from_ty == to_ty => {
                            span_lint(cx,
                                      USELESS_TRANSMUTE,
                                      e.span,
                                      &format!("transmute from a type (`{}`) to itself", from_ty))
                        },
                        (&ty::TyRef(_, rty), &ty::TyRawPtr(ptr_ty)) => {
                            span_lint_and_then(cx,
                                               USELESS_TRANSMUTE,
                                               e.span,
                                               "transmute from a reference to a pointer",
                                               |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                                   let sugg = if ptr_ty == rty {
                                                       arg.as_ty(to_ty)
                                                   } else {
                                                       arg.as_ty(cx.tcx.mk_ptr(rty)).as_ty(to_ty)
                                                   };

                                                   db.span_suggestion(e.span, "try", sugg.to_string());
                                               })
                        },
                        (&ty::TyInt(_), &ty::TyRawPtr(_)) |
                        (&ty::TyUint(_), &ty::TyRawPtr(_)) => {
                            span_lint_and_then(cx,
                                               USELESS_TRANSMUTE,
                                               e.span,
                                               "transmute from an integer to a pointer",
                                               |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                                   db.span_suggestion(e.span,
                                                                      "try",
                                                                      arg.as_ty(&to_ty.to_string()).to_string());
                                               })
                        },
                        (&ty::TyFloat(_), &ty::TyRef(..)) |
                        (&ty::TyFloat(_), &ty::TyRawPtr(_)) |
                        (&ty::TyChar, &ty::TyRef(..)) |
                        (&ty::TyChar, &ty::TyRawPtr(_)) => {
                            span_lint(cx,
                                      WRONG_TRANSMUTE,
                                      e.span,
                                      &format!("transmute from a `{}` to a pointer", from_ty))
                        },
                        (&ty::TyRawPtr(from_ptr), _) if from_ptr.ty == to_ty => {
                            span_lint(cx,
                                      CROSSPOINTER_TRANSMUTE,
                                      e.span,
                                      &format!("transmute from a type (`{}`) to the type that it points to (`{}`)",
                                               from_ty,
                                               to_ty))
                        },
                        (_, &ty::TyRawPtr(to_ptr)) if to_ptr.ty == from_ty => {
                            span_lint(cx,
                                      CROSSPOINTER_TRANSMUTE,
                                      e.span,
                                      &format!("transmute from a type (`{}`) to a pointer to that type (`{}`)",
                                               from_ty,
                                               to_ty))
                        },
                        (&ty::TyRawPtr(from_pty), &ty::TyRef(_, to_rty)) => {
                            span_lint_and_then(cx,
                                               TRANSMUTE_PTR_TO_REF,
                                               e.span,
                                               &format!("transmute from a pointer type (`{}`) to a reference type \
                                                         (`{}`)",
                                                        from_ty,
                                                        to_ty),
                                               |db| {
                                let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                let (deref, cast) = if to_rty.mutbl == Mutability::MutMutable {
                                    ("&mut *", "*mut")
                                } else {
                                    ("&*", "*const")
                                };

                                let arg = if from_pty.ty == to_rty.ty {
                                    arg
                                } else {
                                    arg.as_ty(&format!("{} {}", cast, get_type_snippet(cx, qpath, to_rty.ty)))
                                };

                                db.span_suggestion(e.span, "try", sugg::make_unop(deref, arg).to_string());
                            })
                        },
                        _ => return,
                    };
                }
            }
        }
    }
}

/// Get the snippet of `Bar` in `…::transmute<Foo, &Bar>`. If that snippet is not available , use
/// the type's `ToString` implementation. In weird cases it could lead to types with invalid `'_`
/// lifetime, but it should be rare.
fn get_type_snippet(cx: &LateContext, path: &QPath, to_rty: Ty) -> String {
    let seg = last_path_segment(path);
    if_let_chain!{[
        let PathParameters::AngleBracketedParameters(ref ang) = seg.parameters,
        let Some(to_ty) = ang.types.get(1),
        let TyRptr(_, ref to_ty) = to_ty.node,
    ], {
        return snippet(cx, to_ty.ty.span, &to_rty.to_string()).to_string();
    }}

    to_rty.to_string()
}
