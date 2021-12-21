//! Helpers used to print a message and abort in case of certain panics and some detected UB.

use crate::prelude::*;

fn codegen_print(fx: &mut FunctionCx<'_, '_, '_>, msg: &str) {
    let puts = fx
        .module
        .declare_function(
            "puts",
            Linkage::Import,
            &Signature {
                call_conv: fx.target_config.default_call_conv,
                params: vec![AbiParam::new(fx.pointer_type)],
                returns: vec![AbiParam::new(types::I32)],
            },
        )
        .unwrap();
    let puts = fx.module.declare_func_in_func(puts, &mut fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(puts, "puts");
    }

    let real_msg = format!("trap at {:?} ({}): {}\0", fx.instance, fx.symbol_name, msg);
    let msg_ptr = fx.anonymous_str(&real_msg);
    fx.bcx.ins().call(puts, &[msg_ptr]);
}

/// Trap code: user1
pub(crate) fn trap_abort(fx: &mut FunctionCx<'_, '_, '_>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    fx.bcx.ins().trap(TrapCode::User(1));
}

/// Use this for example when a function call should never return. This will fill the current block,
/// so you can **not** add instructions to it afterwards.
///
/// Trap code: user65535
pub(crate) fn trap_unreachable(fx: &mut FunctionCx<'_, '_, '_>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
}

/// Like `trap_unreachable` but returns a fake value of the specified type.
///
/// Trap code: user65535
pub(crate) fn trap_unreachable_ret_value<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    dest_layout: TyAndLayout<'tcx>,
    msg: impl AsRef<str>,
) -> CValue<'tcx> {
    codegen_print(fx, msg.as_ref());
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::UnreachableCodeReached);
    CValue::by_ref(Pointer::const_addr(fx, 0), dest_layout)
}

/// Use this when something is unimplemented, but `libcore` or `libstd` requires it to codegen.
/// Unlike `trap_unreachable` this will not fill the current block, so you **must** add instructions
/// to it afterwards.
///
/// Trap code: user65535
pub(crate) fn trap_unimplemented(fx: &mut FunctionCx<'_, '_, '_>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::User(!0));
}

/// Like `trap_unimplemented` but returns a fake value of the specified type.
///
/// Trap code: user65535
pub(crate) fn trap_unimplemented_ret_value<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    dest_layout: TyAndLayout<'tcx>,
    msg: impl AsRef<str>,
) -> CValue<'tcx> {
    trap_unimplemented(fx, msg);
    CValue::by_ref(Pointer::const_addr(fx, 0), dest_layout)
}
