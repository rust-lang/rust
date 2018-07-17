//! lint on C-like enums that are `repr(isize/usize)` and have values that
//! don't fit into an `i32`

use rustc::lint::*;
use rustc::hir::*;
use rustc::ty;
use rustc::ty::subst::Substs;
use syntax::ast::{IntTy, UintTy};
use crate::utils::span_lint;
use crate::consts::{Constant, miri_to_const};
use rustc::ty::util::IntTypeExt;
use rustc::mir::interpret::GlobalId;

/// **What it does:** Checks for C-like enumerations that are
/// `repr(isize/usize)` and have values that don't fit into an `i32`.
///
/// **Why is this bad?** This will truncate the variant value on 32 bit
/// architectures, but works fine on 64 bit.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// #[repr(usize)]
/// enum NonPortable {
///     X = 0x1_0000_0000,
///     Y = 0
/// }
/// ```
declare_clippy_lint! {
    pub ENUM_CLIKE_UNPORTABLE_VARIANT,
    correctness,
    "C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`"
}

pub struct UnportableVariant;

impl LintPass for UnportableVariant {
    fn get_lints(&self) -> LintArray {
        lint_array!(ENUM_CLIKE_UNPORTABLE_VARIANT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnportableVariant {
    #[allow(cast_possible_truncation, cast_sign_loss)]
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if cx.tcx.data_layout.pointer_size.bits() != 64 {
            return;
        }
        if let ItemKind::Enum(ref def, _) = item.node {
            for var in &def.variants {
                let variant = &var.node;
                if let Some(ref anon_const) = variant.disr_expr {
                    let param_env = ty::ParamEnv::empty();
                    let did = cx.tcx.hir.body_owner_def_id(anon_const.body);
                    let substs = Substs::identity_for_item(cx.tcx.global_tcx(), did);
                    let instance = ty::Instance::new(did, substs);
                    let cid = GlobalId {
                        instance,
                        promoted: None
                    };
                    let constant = cx.tcx.const_eval(param_env.and(cid)).ok();
                    if let Some(Constant::Int(val)) = constant.and_then(|c| miri_to_const(cx.tcx, c)) {
                        let mut ty = cx.tcx.type_of(did);
                        if let ty::TyAdt(adt, _) = ty.sty {
                            if adt.is_enum() {
                                ty = adt.repr.discr_type().to_ty(cx.tcx);
                            }
                        }
                        match ty.sty {
                            ty::TyInt(IntTy::Isize) => {
                                let val = ((val as i128) << 64) >> 64;
                                if val <= i128::from(i32::max_value()) && val >= i128::from(i32::min_value()) {
                                    continue;
                                }
                            }
                            ty::TyUint(UintTy::Usize) if val > u128::from(u32::max_value()) => {},
                            _ => continue,
                        }
                        span_lint(
                            cx,
                            ENUM_CLIKE_UNPORTABLE_VARIANT,
                            var.span,
                            "Clike enum variant discriminant is not portable to 32-bit targets",
                        );
                    };
                }
            }
        }
    }
}
