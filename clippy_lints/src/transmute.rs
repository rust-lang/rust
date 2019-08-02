use crate::utils::{last_path_segment, match_def_path, paths, snippet, span_lint, span_lint_and_then, sugg};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::{self, Ty};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::borrow::Cow;
use syntax::ast;

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes that can't ever be correct on any
    /// architecture.
    ///
    /// **Why is this bad?** It's basically guaranteed to be undefined behaviour.
    ///
    /// **Known problems:** When accessing C, users might want to store pointer
    /// sized objects in `extradata` arguments to save an allocation.
    ///
    /// **Example:**
    /// ```ignore
    /// let ptr: *const T = core::intrinsics::transmute('x')
    /// ```
    pub WRONG_TRANSMUTE,
    correctness,
    "transmutes that are confusing at best, undefined behaviour at worst and always useless"
}

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes to the original type of the object
    /// and transmutes that could be a cast.
    ///
    /// **Why is this bad?** Readability. The code tricks people into thinking that
    /// something complex is going on.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// core::intrinsics::transmute(t); // where the result type is the same as `t`'s
    /// ```
    pub USELESS_TRANSMUTE,
    complexity,
    "transmutes that have the same to and from types or could be a cast/coercion"
}

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes between a type `T` and `*T`.
    ///
    /// **Why is this bad?** It's easy to mistakenly transmute between a type and a
    /// pointer to that type.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// core::intrinsics::transmute(t) // where the result type is the same as
    ///                                // `*t` or `&t`'s
    /// ```
    pub CROSSPOINTER_TRANSMUTE,
    complexity,
    "transmutes that have to or from types that are a pointer to the other"
}

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes from a pointer to a reference.
    ///
    /// **Why is this bad?** This can always be rewritten with `&` and `*`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// unsafe {
    ///     let _: &T = std::mem::transmute(p); // where p: *const T
    /// }
    ///
    /// // can be written:
    /// let _: &T = &*p;
    /// ```
    pub TRANSMUTE_PTR_TO_REF,
    complexity,
    "transmutes from a pointer to a reference type"
}

declare_clippy_lint! {
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
    /// let x = 1_u32;
    /// unsafe {
    ///     let _: char = std::mem::transmute(x); // where x: u32
    /// }
    ///
    /// // should be:
    /// let _ = std::char::from_u32(x).unwrap();
    /// ```
    pub TRANSMUTE_INT_TO_CHAR,
    complexity,
    "transmutes from an integer to a `char`"
}

declare_clippy_lint! {
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
    /// let b: &[u8] = &[1_u8, 2_u8];
    /// unsafe {
    ///     let _: &str = std::mem::transmute(b); // where b: &[u8]
    /// }
    ///
    /// // should be:
    /// let _ = std::str::from_utf8(b).unwrap();
    /// ```
    pub TRANSMUTE_BYTES_TO_STR,
    complexity,
    "transmutes from a `&[u8]` to a `&str`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes from an integer to a `bool`.
    ///
    /// **Why is this bad?** This might result in an invalid in-memory representation of a `bool`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = 1_u8;
    /// unsafe {
    ///     let _: bool = std::mem::transmute(x); // where x: u8
    /// }
    ///
    /// // should be:
    /// let _: bool = x != 0;
    /// ```
    pub TRANSMUTE_INT_TO_BOOL,
    complexity,
    "transmutes from an integer to a `bool`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for transmutes from an integer to a float.
    ///
    /// **Why is this bad?** Transmutes are dangerous and error-prone, whereas `from_bits` is intuitive
    /// and safe.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// unsafe {
    ///     let _: f32 = std::mem::transmute(1_u32); // where x: u32
    /// }
    ///
    /// // should be:
    /// let _: f32 = f32::from_bits(1_u32);
    /// ```
    pub TRANSMUTE_INT_TO_FLOAT,
    complexity,
    "transmutes from an integer to a float"
}

declare_clippy_lint! {
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
    /// let _ = ptr as *const f32;
    /// let _ = unsafe{ &*(&1u32 as *const u32 as *const f32) };
    /// ```
    pub TRANSMUTE_PTR_TO_PTR,
    complexity,
    "transmutes from a pointer to a pointer / a reference to a reference"
}

declare_lint_pass!(Transmute => [
    CROSSPOINTER_TRANSMUTE,
    TRANSMUTE_PTR_TO_REF,
    TRANSMUTE_PTR_TO_PTR,
    USELESS_TRANSMUTE,
    WRONG_TRANSMUTE,
    TRANSMUTE_INT_TO_CHAR,
    TRANSMUTE_BYTES_TO_STR,
    TRANSMUTE_INT_TO_BOOL,
    TRANSMUTE_INT_TO_FLOAT,
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Transmute {
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprKind::Call(ref path_expr, ref args) = e.node {
            if let ExprKind::Path(ref qpath) = path_expr.node {
                if let Some(def_id) = cx.tables.qpath_res(qpath, path_expr.hir_id).opt_def_id() {
                    if match_def_path(cx, def_id, &paths::TRANSMUTE) {
                        let from_ty = cx.tables.expr_ty(&args[0]);
                        let to_ty = cx.tables.expr_ty(e);

                        match (&from_ty.sty, &to_ty.sty) {
                            _ if from_ty == to_ty => span_lint(
                                cx,
                                USELESS_TRANSMUTE,
                                e.span,
                                &format!("transmute from a type (`{}`) to itself", from_ty),
                            ),
                            (&ty::Ref(_, rty, rty_mutbl), &ty::RawPtr(ptr_ty)) => span_lint_and_then(
                                cx,
                                USELESS_TRANSMUTE,
                                e.span,
                                "transmute from a reference to a pointer",
                                |db| {
                                    if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                        let rty_and_mut = ty::TypeAndMut {
                                            ty: rty,
                                            mutbl: rty_mutbl,
                                        };

                                        let sugg = if ptr_ty == rty_and_mut {
                                            arg.as_ty(to_ty)
                                        } else {
                                            arg.as_ty(cx.tcx.mk_ptr(rty_and_mut)).as_ty(to_ty)
                                        };

                                        db.span_suggestion(e.span, "try", sugg.to_string(), Applicability::Unspecified);
                                    }
                                },
                            ),
                            (&ty::Int(_), &ty::RawPtr(_)) | (&ty::Uint(_), &ty::RawPtr(_)) => span_lint_and_then(
                                cx,
                                USELESS_TRANSMUTE,
                                e.span,
                                "transmute from an integer to a pointer",
                                |db| {
                                    if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                        db.span_suggestion(
                                            e.span,
                                            "try",
                                            arg.as_ty(&to_ty.to_string()).to_string(),
                                            Applicability::Unspecified,
                                        );
                                    }
                                },
                            ),
                            (&ty::Float(_), &ty::Ref(..))
                            | (&ty::Float(_), &ty::RawPtr(_))
                            | (&ty::Char, &ty::Ref(..))
                            | (&ty::Char, &ty::RawPtr(_)) => span_lint(
                                cx,
                                WRONG_TRANSMUTE,
                                e.span,
                                &format!("transmute from a `{}` to a pointer", from_ty),
                            ),
                            (&ty::RawPtr(from_ptr), _) if from_ptr.ty == to_ty => span_lint(
                                cx,
                                CROSSPOINTER_TRANSMUTE,
                                e.span,
                                &format!(
                                    "transmute from a type (`{}`) to the type that it points to (`{}`)",
                                    from_ty, to_ty
                                ),
                            ),
                            (_, &ty::RawPtr(to_ptr)) if to_ptr.ty == from_ty => span_lint(
                                cx,
                                CROSSPOINTER_TRANSMUTE,
                                e.span,
                                &format!(
                                    "transmute from a type (`{}`) to a pointer to that type (`{}`)",
                                    from_ty, to_ty
                                ),
                            ),
                            (&ty::RawPtr(from_pty), &ty::Ref(_, to_ref_ty, mutbl)) => span_lint_and_then(
                                cx,
                                TRANSMUTE_PTR_TO_REF,
                                e.span,
                                &format!(
                                    "transmute from a pointer type (`{}`) to a reference type \
                                     (`{}`)",
                                    from_ty, to_ty
                                ),
                                |db| {
                                    let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                    let (deref, cast) = if mutbl == Mutability::MutMutable {
                                        ("&mut *", "*mut")
                                    } else {
                                        ("&*", "*const")
                                    };

                                    let arg = if from_pty.ty == to_ref_ty {
                                        arg
                                    } else {
                                        arg.as_ty(&format!("{} {}", cast, get_type_snippet(cx, qpath, to_ref_ty)))
                                    };

                                    db.span_suggestion(
                                        e.span,
                                        "try",
                                        sugg::make_unop(deref, arg).to_string(),
                                        Applicability::Unspecified,
                                    );
                                },
                            ),
                            (&ty::Int(ast::IntTy::I32), &ty::Char) | (&ty::Uint(ast::UintTy::U32), &ty::Char) => {
                                span_lint_and_then(
                                    cx,
                                    TRANSMUTE_INT_TO_CHAR,
                                    e.span,
                                    &format!("transmute from a `{}` to a `char`", from_ty),
                                    |db| {
                                        let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                        let arg = if let ty::Int(_) = from_ty.sty {
                                            arg.as_ty(ast::UintTy::U32)
                                        } else {
                                            arg
                                        };
                                        db.span_suggestion(
                                            e.span,
                                            "consider using",
                                            format!("std::char::from_u32({}).unwrap()", arg.to_string()),
                                            Applicability::Unspecified,
                                        );
                                    },
                                )
                            },
                            (&ty::Ref(_, ty_from, from_mutbl), &ty::Ref(_, ty_to, to_mutbl)) => {
                                if_chain! {
                                    if let (&ty::Slice(slice_ty), &ty::Str) = (&ty_from.sty, &ty_to.sty);
                                    if let ty::Uint(ast::UintTy::U8) = slice_ty.sty;
                                    if from_mutbl == to_mutbl;
                                    then {
                                        let postfix = if from_mutbl == Mutability::MutMutable {
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
                                                    Applicability::Unspecified,
                                                );
                                            }
                                        )
                                    } else {
                                        if cx.tcx.erase_regions(&from_ty) != cx.tcx.erase_regions(&to_ty) {
                                            span_lint_and_then(
                                                cx,
                                                TRANSMUTE_PTR_TO_PTR,
                                                e.span,
                                                "transmute from a reference to a reference",
                                                |db| if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                                    let ty_from_and_mut = ty::TypeAndMut {
                                                        ty: ty_from,
                                                        mutbl: from_mutbl
                                                    };
                                                    let ty_to_and_mut = ty::TypeAndMut { ty: ty_to, mutbl: to_mutbl };
                                                    let sugg_paren = arg
                                                        .as_ty(cx.tcx.mk_ptr(ty_from_and_mut))
                                                        .as_ty(cx.tcx.mk_ptr(ty_to_and_mut));
                                                    let sugg = if to_mutbl == Mutability::MutMutable {
                                                        sugg_paren.mut_addr_deref()
                                                    } else {
                                                        sugg_paren.addr_deref()
                                                    };
                                                    db.span_suggestion(
                                                        e.span,
                                                        "try",
                                                        sugg.to_string(),
                                                        Applicability::Unspecified,
                                                    );
                                                },
                                            )
                                        }
                                    }
                                }
                            },
                            (&ty::RawPtr(_), &ty::RawPtr(to_ty)) => span_lint_and_then(
                                cx,
                                TRANSMUTE_PTR_TO_PTR,
                                e.span,
                                "transmute from a pointer to a pointer",
                                |db| {
                                    if let Some(arg) = sugg::Sugg::hir_opt(cx, &args[0]) {
                                        let sugg = arg.as_ty(cx.tcx.mk_ptr(to_ty));
                                        db.span_suggestion(e.span, "try", sugg.to_string(), Applicability::Unspecified);
                                    }
                                },
                            ),
                            (&ty::Int(ast::IntTy::I8), &ty::Bool) | (&ty::Uint(ast::UintTy::U8), &ty::Bool) => {
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
                                            Applicability::Unspecified,
                                        );
                                    },
                                )
                            },
                            (&ty::Int(_), &ty::Float(_)) | (&ty::Uint(_), &ty::Float(_)) => span_lint_and_then(
                                cx,
                                TRANSMUTE_INT_TO_FLOAT,
                                e.span,
                                &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                                |db| {
                                    let arg = sugg::Sugg::hir(cx, &args[0], "..");
                                    let arg = if let ty::Int(int_ty) = from_ty.sty {
                                        arg.as_ty(format!(
                                            "u{}",
                                            int_ty.bit_width().map_or_else(|| "size".to_string(), |v| v.to_string())
                                        ))
                                    } else {
                                        arg
                                    };
                                    db.span_suggestion(
                                        e.span,
                                        "consider using",
                                        format!("{}::from_bits({})", to_ty, arg.to_string()),
                                        Applicability::Unspecified,
                                    );
                                },
                            ),
                            _ => return,
                        };
                    }
                }
            }
        }
    }
}

/// Gets the snippet of `Bar` in `…::transmute<Foo, &Bar>`. If that snippet is
/// not available , use
/// the type's `ToString` implementation. In weird cases it could lead to types
/// with invalid `'_`
/// lifetime, but it should be rare.
fn get_type_snippet(cx: &LateContext<'_, '_>, path: &QPath, to_ref_ty: Ty<'_>) -> String {
    let seg = last_path_segment(path);
    if_chain! {
        if let Some(ref params) = seg.args;
        if !params.parenthesized;
        if let Some(to_ty) = params.args.iter().filter_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        }).nth(1);
        if let TyKind::Rptr(_, ref to_ty) = to_ty.node;
        then {
            return snippet(cx, to_ty.ty.span, &to_ref_ty.to_string()).to_string();
        }
    }

    to_ref_ty.to_string()
}
