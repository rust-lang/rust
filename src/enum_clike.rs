//! lint on C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`

use rustc::lint::*;
use syntax::ast::{IntTy, UintTy};
use syntax::attr::*;
use rustc_front::hir::*;
use rustc::middle::const_eval::{ConstVal, EvalHint, eval_const_expr_partial};
use rustc::middle::ty;
use utils::span_lint;

/// **What it does:** Lints on C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`.
///
/// **Why is this bad?** This will truncate the variant value on 32bit architectures, but works fine on 64 bit.
///
/// **Known problems:** None
///
/// **Example:** `#[repr(usize)] enum NonPortable { X = 0x1_0000_0000, Y = 0 }`
declare_lint! {
    pub ENUM_CLIKE_UNPORTABLE_VARIANT, Warn,
    "finds C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`"
}

pub struct EnumClikeUnportableVariant;

impl LintPass for EnumClikeUnportableVariant {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_CLIKE_UNPORTABLE_VARIANT)
    }
}

impl LateLintPass for EnumClikeUnportableVariant {
    #[allow(cast_possible_truncation, cast_sign_loss)]
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemEnum(ref def, _) = item.node {
            for var in &def.variants {
                let variant = &var.node;
                if let Some(ref disr) = variant.disr_expr {
                    let cv = eval_const_expr_partial(cx.tcx, &**disr, EvalHint::ExprTypeChecked, None);
                    let bad = match (cv, &cx.tcx.expr_ty(&**disr).sty) {
                        (Ok(ConstVal::Int(i)), &ty::TyInt(IntTy::Is)) => i as i32 as i64 != i,
                        (Ok(ConstVal::Uint(i)), &ty::TyInt(IntTy::Is)) => i as i32 as u64 != i,
                        (Ok(ConstVal::Int(i)), &ty::TyUint(UintTy::Us)) => (i < 0) || (i as u32 as i64 != i),
                        (Ok(ConstVal::Uint(i)), &ty::TyUint(UintTy::Us)) => i as u32 as u64 != i,
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
