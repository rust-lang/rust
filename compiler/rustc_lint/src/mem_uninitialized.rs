use crate::context::LintContext;
use crate::LateContext;
use crate::LateLintPass;
use rustc_hir as hir;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::Ty;
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::abi::VariantIdx;
use std::fmt::Write;

declare_lint! {
    /// The `mem_uninitialized` lint detects all uses of `std::mem::uninitialized` that are not
    /// known to be safe.
    ///
    /// This function is extremely dangerous, and nearly all uses of it cause immediate Undefined
    /// Behavior.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(mem_uninitialized)]
    /// fn main() {
    ///     let x: [char; 16] = unsafe { std::mem::uninitialized() };
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating an invalid value is undefined behavior, and nearly all types are invalid when left
    /// uninitialized.
    ///
    /// To avoid churn, however, this will not lint for types made up entirely of integers, floats,
    /// or raw pointers. This is not saying that leaving these types uninitialized is okay,
    /// however.
    pub MEM_UNINITIALIZED,
    Warn,
    "use of mem::uninitialized",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "FIXME: fill this in",
        reason: FutureIncompatibilityReason::FutureReleaseErrorReportNow,
        explain_reason: false,
    };
}

declare_lint_pass!(MemUninitialized => [MEM_UNINITIALIZED]);

/// Information about why a type cannot be initialized this way.
/// Contains an error message and optionally a span to point at.
pub struct InitError {
    msg: String,
    span: Option<Span>,
    generic: bool,
}

impl InitError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into(), span: None, generic: false }
    }

    fn with_span(msg: impl Into<String>, span: Span) -> Self {
        Self { msg: msg.into(), span: Some(span), generic: false }
    }

    fn generic() -> Self {
        Self {
            msg: "type might not be allowed to be left uninitialized".to_string(),
            span: None,
            generic: true,
        }
    }
}

/// Return `None` only if we are sure this type does
/// allow being left uninitialized.
pub fn ty_find_init_error<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<InitError> {
    use rustc_type_ir::sty::TyKind::*;
    match ty.kind() {
        // Primitive types that don't like 0 as a value.
        Ref(..) => Some(InitError::new("references must be non-null")),
        Adt(..) if ty.is_box() => Some(InitError::new("`Box` must be non-null")),
        FnPtr(..) => Some(InitError::new("function pointers must be non-null")),
        Never => Some(InitError::new("the `!` type has no valid value")),
        RawPtr(tm) if matches!(tm.ty.kind(), Dynamic(..)) =>
        // raw ptr to dyn Trait
        {
            Some(InitError::new("the vtable of a wide raw pointer must be non-null"))
        }
        // Primitive types with other constraints.
        Bool => Some(InitError::new("booleans must be either `true` or `false`")),
        Char => Some(InitError::new("characters must be a valid Unicode codepoint")),
        Adt(adt_def, _) if adt_def.is_union() => None,
        // Recurse and checks for some compound types.
        Adt(adt_def, substs) => {
            // First check if this ADT has a layout attribute (like `NonNull` and friends).
            use std::ops::Bound;
            match cx.tcx.layout_scalar_valid_range(adt_def.did()) {
                // We exploit here that `layout_scalar_valid_range` will never
                // return `Bound::Excluded`.  (And we have tests checking that we
                // handle the attribute correctly.)
                (Bound::Included(lo), _) if lo > 0 => {
                    return Some(InitError::new(format!("`{ty}` must be non-null")));
                }
                (Bound::Included(_), _) | (_, Bound::Included(_)) => {
                    return Some(InitError::new(format!(
                        "`{ty}` must be initialized inside its custom valid range"
                    )));
                }
                _ => {}
            }
            // Now, recurse.
            match adt_def.variants().len() {
                0 => Some(InitError::new("enums with no variants have no valid value")),
                1 => {
                    // Struct, or enum with exactly one variant.
                    // Proceed recursively, check all fields.
                    let variant = &adt_def.variant(VariantIdx::from_u32(0));
                    variant.fields.iter().find_map(|field| {
                        ty_find_init_error(cx, field.ty(cx.tcx, substs)).map(
                            |InitError { mut msg, span, generic }| {
                                if span.is_none() {
                                    // Point to this field, should be helpful for figuring
                                    // out where the source of the error is.
                                    let span = cx.tcx.def_span(field.did);
                                    write!(&mut msg, " (in this {} field)", adt_def.descr())
                                        .unwrap();

                                    InitError { msg, span: Some(span), generic }
                                } else {
                                    // Just forward.
                                    InitError { msg, span, generic }
                                }
                            },
                        )
                    })
                }
                // Multi-variant enum.
                _ => {
                    // This will warn on something like Result<MaybeUninit<u32>, !> which
                    // is not UB under the current enum layout, even ignoring the 0x01
                    // filling.
                    //
                    // That's probably fine though.
                    let span = cx.tcx.def_span(adt_def.did());
                    Some(InitError::with_span("enums have to be initialized to a variant", span))
                }
            }
        }
        Tuple(..) => {
            // Proceed recursively, check all fields.
            ty.tuple_fields().iter().find_map(|field| ty_find_init_error(cx, field))
        }
        Array(ty, len) => {
            match len.try_eval_usize(cx.tcx, cx.param_env) {
                // Array known to be zero sized, we can't warn.
                Some(0) => None,

                // Array length known to be nonzero, warn.
                Some(1..) => ty_find_init_error(cx, *ty),

                // Array length unknown, use the "might not permit" wording.
                None => ty_find_init_error(cx, *ty).map(|mut e| {
                    e.generic = true;
                    e
                }),
            }
        }
        Int(_) | Uint(_) | Float(_) | RawPtr(_) => {
            // These are Plain Old Data types that people expect to work if they leave them
            // uninitialized.
            None
        }
        // Pessimistic fallback.
        _ => Some(InitError::generic()),
    }
}

impl<'tcx> LateLintPass<'tcx> for MemUninitialized {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'_>) {
        /// Determine if this expression is a "dangerous initialization".
        fn is_dangerous_init(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
            if let hir::ExprKind::Call(ref path_expr, _) = expr.kind {
                // Find calls to `mem::{uninitialized,zeroed}` methods.
                if let hir::ExprKind::Path(ref qpath) = path_expr.kind {
                    if let Some(def_id) = cx.qpath_res(qpath, path_expr.hir_id).opt_def_id() {
                        if cx.tcx.is_diagnostic_item(sym::mem_uninitialized, def_id) {
                            return true;
                        }
                    }
                }
            }

            false
        }

        if is_dangerous_init(cx, expr) {
            // This conjures an instance of a type out of nothing,
            // using zeroed or uninitialized memory.
            // We are extremely conservative with what we warn about.
            let conjured_ty = cx.typeck_results().expr_ty(expr);
            if let Some(init_error) = with_no_trimmed_paths!(ty_find_init_error(cx, conjured_ty)) {
                let main_msg = with_no_trimmed_paths!(if init_error.generic {
                    format!(
                        "the type `{conjured_ty}` is generic, and might not permit being left uninitialized"
                    )
                } else {
                    format!("the type `{conjured_ty}` does not permit being left uninitialized")
                });

                // FIXME(davidtwco): make translatable
                cx.struct_span_lint(MEM_UNINITIALIZED, expr.span, |lint| {
                    let mut err = lint.build(&main_msg);

                    err.span_label(expr.span, "this code causes undefined behavior when executed");
                    err.span_label(
                        expr.span,
                        "help: use `MaybeUninit<T>` instead, \
                            and only call `assume_init` after initialization is done",
                    );
                    if let Some(span) = init_error.span {
                        err.span_note(span, &init_error.msg);
                    } else {
                        err.note(&init_error.msg);
                    }
                    err.emit();
                });
            }
        }
    }
}
