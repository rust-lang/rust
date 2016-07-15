//! lint on C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`

use rustc::lint::*;
use rustc::middle::const_val::ConstVal;
use rustc_const_math::*;
use rustc::hir::*;
use utils::span_lint;

/// **What it does:** Lints on C-like enumerations that are `repr(isize/usize)` and have values
/// that don't fit into an `i32`.
///
/// **Why is this bad?** This will truncate the variant value on 32 bit architectures, but works
/// fine on 64 bit.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// #[repr(usize)]
/// enum NonPortable {
///     X = 0x1_0000_0000,
///     Y = 0
/// }
/// ```
declare_lint! {
    pub ENUM_CLIKE_UNPORTABLE_VARIANT, Warn,
    "finds C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`"
}

pub struct UnportableVariant;

impl LintPass for UnportableVariant {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_CLIKE_UNPORTABLE_VARIANT)
    }
}

impl LateLintPass for UnportableVariant {
    #[allow(cast_possible_truncation, cast_sign_loss)]
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemEnum(ref def, _) = item.node {
            for var in &def.variants {
                let variant = &var.node;
                if let Some(ref disr) = variant.disr_expr {
                    use rustc_const_eval::*;
                    let bad = match eval_const_expr_partial(cx.tcx, &**disr, EvalHint::ExprTypeChecked, None) {
                        Ok(ConstVal::Integral(Usize(Us64(i)))) => i as u32 as u64 != i,
                        Ok(ConstVal::Integral(Isize(Is64(i)))) => i as i32 as i64 != i,
                        _ => false,
                    };
                    if bad {
                        span_lint(cx,
                                  ENUM_CLIKE_UNPORTABLE_VARIANT,
                                  var.span,
                                  "Clike enum variant discriminant is not portable to 32-bit targets");
                    }
                }
            }
        }
    }
}
