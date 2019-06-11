use crate::prelude::*;

fn codegen_print(fx: &mut FunctionCx<'_, '_, impl cranelift_module::Backend>, msg: &str) {
    let puts = fx.module.declare_function("puts", Linkage::Import, &Signature {
        call_conv: CallConv::SystemV,
        params: vec![AbiParam::new(pointer_ty(fx.tcx))],
        returns: vec![],
    }).unwrap();
    let puts = fx.module.declare_func_in_func(puts, &mut fx.bcx.func);

    let symbol_name = fx.tcx.symbol_name(fx.instance);
    let msg_bytes = format!("trap at {:?} ({}): {}\0", fx.instance, symbol_name, msg).into_bytes().into_boxed_slice();
    let mut data_ctx = DataContext::new();
    data_ctx.define(msg_bytes);
    let msg_id = fx.module.declare_data(&(symbol_name.as_str().to_string() + msg), Linkage::Local, false, None).unwrap();

    // Ignore DuplicateDefinition error, as the data will be the same
    let _ = fx.module.define_data(msg_id, &data_ctx);

    let local_msg_id = fx.module.declare_data_in_func(msg_id, fx.bcx.func);
    let msg_ptr = fx.bcx.ins().global_value(pointer_ty(fx.tcx), local_msg_id);
    fx.bcx.ins().call(puts, &[msg_ptr]);
}

/// Trap code: user0
pub fn trap_panic(fx: &mut FunctionCx<'_, '_, impl cranelift_module::Backend>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    fx.bcx.ins().trap(TrapCode::User(0));
}

/// Trap code: user65535
pub fn trap_unreachable(fx: &mut FunctionCx<'_, '_, impl cranelift_module::Backend>, msg: impl AsRef<str>) {
    codegen_print(fx, msg.as_ref());
    fx.bcx.ins().trap(TrapCode::User(!0));
}

/// Trap code: user65535
pub fn trap_unreachable_ret_value<'tcx>(fx: &mut FunctionCx<'_, 'tcx, impl cranelift_module::Backend>, dest_layout: TyLayout<'tcx>, msg: impl AsRef<str>) -> CValue<'tcx> {
    codegen_print(fx, msg.as_ref());
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::User(!0));
    let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
    CValue::by_ref(zero, dest_layout)
}

/// Trap code: user65535
pub fn trap_unreachable_ret_place<'tcx>(fx: &mut FunctionCx<'_, 'tcx, impl cranelift_module::Backend>, dest_layout: TyLayout<'tcx>, msg: impl AsRef<str>) -> CPlace<'tcx> {
    codegen_print(fx, msg.as_ref());
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::User(!0));
    let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
    CPlace::Addr(zero, None, dest_layout)
}
