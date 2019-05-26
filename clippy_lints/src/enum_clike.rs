//! lint on C-like enums that are `repr(isize/usize)` and have values that
//! don't fit into an `i32`

use crate::consts::{miri_to_const, Constant};
use crate::utils::span_lint;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::mir::interpret::GlobalId;
use rustc::ty;
use rustc::ty::subst::InternalSubsts;
use rustc::ty::util::IntTypeExt;
use rustc::{declare_lint_pass, declare_tool_lint};
use std::convert::TryFrom;
use syntax::ast::{IntTy, UintTy};

declare_clippy_lint! {
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
    ///     Y = 0,
    /// }
    /// ```
    pub ENUM_CLIKE_UNPORTABLE_VARIANT,
    correctness,
    "C-like enums that are `repr(isize/usize)` and have values that don't fit into an `i32`"
}

declare_lint_pass!(UnportableVariant => [ENUM_CLIKE_UNPORTABLE_VARIANT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnportableVariant {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if cx.tcx.data_layout.pointer_size.bits() != 64 {
            return;
        }
        if let ItemKind::Enum(ref def, _) = item.node {
            for var in &def.variants {
                let variant = &var.node;
                if let Some(ref anon_const) = variant.disr_expr {
                    let param_env = ty::ParamEnv::empty();
                    let def_id = cx.tcx.hir().body_owner_def_id(anon_const.body);
                    let substs = InternalSubsts::identity_for_item(cx.tcx.global_tcx(), def_id);
                    let instance = ty::Instance::new(def_id, substs);
                    let c_id = GlobalId {
                        instance,
                        promoted: None,
                    };
                    let constant = cx.tcx.const_eval(param_env.and(c_id)).ok();
                    if let Some(Constant::Int(val)) = constant.and_then(miri_to_const) {
                        let mut ty = cx.tcx.type_of(def_id);
                        if let ty::Adt(adt, _) = ty.sty {
                            if adt.is_enum() {
                                ty = adt.repr.discr_type().to_ty(cx.tcx);
                            }
                        }
                        match ty.sty {
                            ty::Int(IntTy::Isize) => {
                                let val = ((val as i128) << 64) >> 64;
                                if i32::try_from(val).is_ok() {
                                    continue;
                                }
                            },
                            ty::Uint(UintTy::Usize) if val > u128::from(u32::max_value()) => {},
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
