use rustc::lint::*;
use rustc::ty::{self, Ty};
use rustc::hir::*;
use std::borrow::Cow;
use syntax::ast;
use utils::{last_path_segment, match_def_path, paths, snippet, span_lint, span_lint_and_then};
use utils::{opt_def_id, sugg};

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
/// let ptr: *const T = core::intrinsics::transmute('x')
/// ```
declare_clippy_lint! {
    pub WRONG_TRANSMUTE,
    correctness,
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
declare_clippy_lint! {
    pub USELESS_TRANSMUTE,
    complexity,
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
/// core::intrinsics::transmute(t) // where the result type is the same as
/// // `*t` or `&t`'s
/// ```
declare_clippy_lint! {
    pub CROSSPOINTER_TRANSMUTE,
    complexity,
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
declare_clippy_lint! {
    pub TRANSMUTE_PTR_TO_REF,
    complexity,
    "transmutes from a pointer to a reference type"
}

/// **What it does:** Checks for transmutes from an integer to a `char`.
///
/// **Why is this bad?** Not every integer is a Unicode scalar value.
///
/// **Known problems:**
/// - [`from_u32`] which this lint suggests using is slower than `transmute`
/// as it needs to validate the input.
/// If you are certain that the input is always a valid Unicode scalar value,
/// use [`from_u32_unchecked`] which is as fast as `transmute`
/// but has a semantically meaningful name.
/// - You might want to handle `None` returned from [`from_u32`] instead of calling `unwrap`.
///
/// [`from_u32`]: https://doc.rust-lang.org/std/char/fn.from_u32.html
/// [`from_u32_unchecked`]: https://doc.rust-lang.org/std/char/fn.from_u32_unchecked.html
///
/// **Example:**
/// ```rust
/// let _: char = std::mem::transmute(x); // where x: u32
/// // should be:
/// let _ = std::char::from_u32(x).unwrap();
/// ```
declare_clippy_lint! {
    pub TRANSMUTE_INT_TO_CHAR,
    complexity,
    "transmutes from an integer to a `char`"
}

/// **What it does:** Checks for transmutes from a `&[u8]` to a `&str`.
///
/// **Why is this bad?** Not every byte slice is a valid UTF-8 string.
///
/// **Known problems:**
/// - [`from_utf8`] which this lint suggests using is slower than `transmute`
/// as it needs to validate the input.
/// If you are certain that the input is always a valid UTF-8,
/// use [`from_utf8_unchecked`] which is as fast as `transmute`
/// but has a semantically meaningful name.
/// - You might want to handle errors returned from [`from_utf8`] instead of calling `unwrap`.
///
/// [`from_utf8`]: https://doc.rust-lang.org/std/str/fn.from_utf8.html
/// [`from_utf8_unchecked`]: https://doc.rust-lang.org/std/str/fn.from_utf8_unchecked.html
///
/// **Example:**
/// ```rust
/// let _: &str = std::mem::transmute(b); // where b: &[u8]
/// // should be:
/// let _ = std::str::from_utf8(b).unwrap();
/// ```
declare_clippy_lint! {
    pub TRANSMUTE_BYTES_TO_STR,
    complexity,
    "transmutes from a `&[u8]` to a `&str`"
}

/// **What it does:** Checks for transmutes from an integer to a `bool`.
///
/// **Why is this bad?** This might result in an invalid in-memory representation of a `bool`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let _: bool = std::mem::transmute(x); // where x: u8
/// // should be:
/// let _: bool = x != 0;
/// ```
declare_clippy_lint! {
    pub TRANSMUTE_INT_TO_BOOL,
    complexity,
    "transmutes from an integer to a `bool`"
}

/// **What it does:** Checks for transmutes from an integer to a float.
///
/// **Why is this bad?** This might result in an invalid in-memory representation of a float.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let _: f32 = std::mem::transmute(x); // where x: u32
/// // should be:
/// let _: f32 = f32::from_bits(x);
/// ```
declare_clippy_lint! {
    pub TRANSMUTE_INT_TO_FLOAT,
    complexity,
    "transmutes from an integer to a float"
}

/// **What it does:** Checks for transmutes from a pointer to a pointer, or
/// from a reference to a reference.
///
/// **Why is this bad?** Transmutes are dangerous, and these can instead be
/// written as casts.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let ptr = &1u32 as *const u32;
/// unsafe {
///     // pointer-to-pointer transmute
///     let _: *const f32 = std::mem::transmute(ptr);
///     // ref-ref transmute
///     let _: &f32 = std::mem::transmute(&1u32);
/// }
/// // These can be respectively written:
/// let _ = ptr as *const f32
/// let _ = unsafe{ &*(&1u32 as *const u32 as *const f32) };
/// ```
declare_clippy_lint! {
    pub TRANSMUTE_PTR_TO_PTR,
    complexity,
    "transmutes from a pointer to a reference type"
}

pub struct Transmute;

impl LintPass for Transmute {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            CROSSPOINTER_TRANSMUTE,
            TRANSMUTE_PTR_TO_REF,
            TRANSMUTE_PTR_TO_PTR,
            USELESS_TRANSMUTE,
            WRONG_TRANSMUTE,
            TRANSMUTE_INT_TO_CHAR,
            TRANSMUTE_BYTES_TO_STR,
            TRANSMUTE_INT_TO_BOOL,
            TRANSMUTE_INT_TO_FLOAT,
        )
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Transmute {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(ref qpath) = path_expr.node {
                if let Some(def_id) = opt_def_id(cx.tables.qpath_def(qpath, path_expr.hir_id)) {
                    if match_def_path(cx.tcx, def_id, &paths::TRANSMUTE) {
                        let from_ty = cx.tables.expr_ty(&args[0]);
                        let to_ty = cx.tables.expr_ty(e);

                        match (&from_ty.sty, &to_ty.sty) {
                            _ if from_ty == to_ty => span_lint(
                                cx,
                                USELESS_TRANSMUTE,
                                e.span,
                                &format!("transmute from a type (`{}`) to itself", from_ty),
                            ),
                            (&ty::TyRef(_, rty), &ty::TyRawPtr(ptr_ty)) => span_lint_and_then(
                                cx,
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
                                },
                            ),
                            (&ty::TyInt(_), &ty::TyRawPtr(_)) | (&ty::TyUint(_), &ty::TyRawPtr(_)) => {
                                span_lint_and_then(
                                    cx,
                                    USELESS_TRANSMUTE,
                                    e.span,
                                    "transmute from an integer to a pointer",
                                    |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                        db.span_suggestion(e.span, "try", arg.as_ty(&to_ty.to_string()).to_string());
                                    },
                                )
                            },
                            (&ty::TyFloat(_), &ty::TyRef(..)) |
                            (&ty::TyFloat(_), &ty::TyRawPtr(_)) |
                            (&ty::TyChar, &ty::TyRef(..)) |
                            (&ty::TyChar, &ty::TyRawPtr(_)) => span_lint(
                                cx,
                                WRONG_TRANSMUTE,
                                e.span,
                                &format!("transmute from a `{}` to a pointer", from_ty),
                            ),
                            (&ty::TyRawPtr(from_ptr), _) if from_ptr.ty == to_ty => span_lint(
                                cx,
                                CROSSPOINTER_TRANSMUTE,
                                e.span,
                                &format!(
                                    "transmute from a type (`{}`) to the type that it points to (`{}`)",
                                    from_ty,
                                    to_ty
                                ),
                            ),
                            (_, &ty::TyRawPtr(to_ptr)) if to_ptr.ty == from_ty => span_lint(
                                cx,
                                CROSSPOINTER_TRANSMUTE,
                                e.span,
                                &format!(
                                    "transmute from a type (`{}`) to a pointer to that type (`{}`)",
                                    from_ty,
                                    to_ty
                                ),
                            ),
                            (&ty::TyRawPtr(from_pty), &ty::TyRef(_, to_ref_ty)) => span_lint_and_then(
                                cx,
                                TRANSMUTE_PTR_TO_REF,
                                e.span,
                                &format!(
                                    "transmute from a pointer type (`{}`) to a reference type \
                                     (`{}`)",
                                    from_ty,
                                    to_ty
                                ),
                                |db| {
                                    let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                    let (deref, cast) = if to_ref_ty.mutbl == Mutability::MutMutable {
                                        ("&mut *", "*mut")
                                    } else {
                                        ("&*", "*const")
                                    };

                                    let arg = if from_pty.ty == to_ref_ty.ty {
                                        arg
                                    } else {
                                        arg.as_ty(&format!("{} {}", cast, get_type_snippet(cx, qpath, to_ref_ty.ty)))
                                    };

                                    db.span_suggestion(e.span, "try", sugg::make_unop(deref, arg).to_string());
                                },
                            ),
                            (&ty::TyInt(ast::IntTy::I32), &ty::TyChar) |
                            (&ty::TyUint(ast::UintTy::U32), &ty::TyChar) => span_lint_and_then(
                                cx,
                                TRANSMUTE_INT_TO_CHAR,
                                e.span,
                                &format!("transmute from a `{}` to a `char`", from_ty),
                                |db| {
                                    let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                    let arg = if let ty::TyInt(_) = from_ty.sty {
                                        arg.as_ty(ty::TyUint(ast::UintTy::U32))
                                    } else {
                                        arg
                                    };
                                    db.span_suggestion(
                                        e.span,
                                        "consider using",
                                        format!("std::char::from_u32({}).unwrap()", arg.to_string()),
                                    );
                                },
                            ),
                            (&ty::TyRef(_, ref ref_from), &ty::TyRef(_, ref ref_to)) => {
                                if_chain! {
                                    if let (&ty::TySlice(slice_ty), &ty::TyStr) = (&ref_from.ty.sty, &ref_to.ty.sty);
                                    if let ty::TyUint(ast::UintTy::U8) = slice_ty.sty;
                                    if ref_from.mutbl == ref_to.mutbl;
                                    then {
                                        let postfix = if ref_from.mutbl == Mutability::MutMutable {
                                            "_mut"
                                        } else {
                                            ""
                                        };

                                        span_lint_and_then(
                                            cx,
                                            TRANSMUTE_BYTES_TO_STR,
                                            e.span,
                                            &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                                            |db| {
                                                db.span_suggestion(
                                                    e.span,
                                                    "consider using",
                                                    format!(
                                                        "std::str::from_utf8{}({}).unwrap()",
                                                        postfix,
                                                        snippet(cx, args[0].span, ".."),
                                                    ),
                                                );
                                            }
                                        )
                                    } else {
                                        span_lint_and_then(
                                            cx,
                                            TRANSMUTE_PTR_TO_PTR,
                                            e.span,
                                            "transmute from a reference to a reference",
                                            |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                                let sugg_paren = arg.as_ty(cx.tcx.mk_ptr(*ref_from)).as_ty(cx.tcx.mk_ptr(*ref_to));
                                                let sugg = if ref_to.mutbl == Mutability::MutMutable {
                                                    sugg_paren.mut_addr_deref()
                                                } else {
                                                    sugg_paren.addr_deref()
                                                };
                                                db.span_suggestion(e.span, "try", sugg.to_string());
                                            },
                                        )
                                    }
                                }
                            },
                            (&ty::TyRawPtr(_), &ty::TyRawPtr(to_ty)) => span_lint_and_then(
                                cx,
                                TRANSMUTE_PTR_TO_PTR,
                                e.span,
                                "transmute from a pointer to a pointer",
                                |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                    let sugg = arg.as_ty(cx.tcx.mk_ptr(to_ty));
                                    db.span_suggestion(e.span, "try", sugg.to_string());
                                },
                            ),
                            (&ty::TyInt(ast::IntTy::I8), &ty::TyBool) | (&ty::TyUint(ast::UintTy::U8), &ty::TyBool) => {
                                span_lint_and_then(
                                    cx,
                                    TRANSMUTE_INT_TO_BOOL,
                                    e.span,
                                    &format!("transmute from a `{}` to a `bool`", from_ty),
                                    |db| {
                                        let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                        let zero = sugg::Sugg::NonParen(Cow::from("0"));
                                        db.span_suggestion(
                                            e.span,
                                            "consider using",
                                            sugg::make_binop(ast::BinOpKind::Ne, &arg, &zero).to_string(),
                                        );
                                    },
                                )
                            },
                            (&ty::TyInt(_), &ty::TyFloat(_)) | (&ty::TyUint(_), &ty::TyFloat(_)) => {
                                span_lint_and_then(
                                    cx,
                                    TRANSMUTE_INT_TO_FLOAT,
                                    e.span,
                                    &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                                    |db| {
                                        let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                        let arg = if let ty::TyInt(int_ty) = from_ty.sty {
                                            arg.as_ty(format!(
                                                "u{}",
                                                int_ty
                                                    .bit_width()
                                                    .map_or_else(|| "size".to_string(), |v| v.to_string())
                                            ))
                                        } else {
                                            arg
                                        };
                                        db.span_suggestion(
                                            e.span,
                                            "consider using",
                                            format!("{}::from_bits({})", to_ty, arg.to_string()),
                                        );
                                    },
                                )
                            },
                            _ => return,
                        };
                    }
                }
            }
        }
    }
}

/// Get the snippet of `Bar` in `…::transmute<Foo, &Bar>`. If that snippet is
/// not available , use
/// the type's `ToString` implementation. In weird cases it could lead to types
/// with invalid `'_`
/// lifetime, but it should be rare.
fn get_type_snippet(cx: &LateContext, path: &QPath, to_ref_ty: Ty) -> String {
    let seg = last_path_segment(path);
    if_chain! {
        if let Some(ref params) = seg.parameters;
        if !params.parenthesized;
        if let Some(to_ty) = params.types.get(1);
        if let TyRptr(_, ref to_ty) = to_ty.node;
        then {
            return snippet(cx, to_ty.ty.span, &to_ref_ty.to_string()).to_string();
        }
    }

    to_ref_ty.to_string()
}
