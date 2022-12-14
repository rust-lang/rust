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

/// Use this when something is unimplemented, but `libcore` or `libstd` requires it to codegen.
///
/// Trap code: user65535
pub(crate) fn trap_unimplemented(fx: &mut FunctionCx<'_, '_, '_>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    fx.bcx.ins().trap(TrapCode::User(!0));
}
