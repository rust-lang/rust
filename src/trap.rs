use crate::prelude::*;

/// Trap code: user0
pub fn trap_panic(bcx: &mut FunctionBuilder) {
    bcx.ins().trap(TrapCode::User(0));
}

/// Trap code: user65535
pub fn trap_unreachable(bcx: &mut FunctionBuilder) {
    bcx.ins().trap(TrapCode::User(!0));
}

pub fn trap_unreachable_ret_value<'tcx>(fx: &mut FunctionCx<'_, 'tcx, impl cranelift_module::Backend>, dest_layout: TyLayout<'tcx>) -> CValue<'tcx> {
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::User(!0));
    let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
    CValue::ByRef(zero, dest_layout)
}

pub fn trap_unreachable_ret_place<'tcx>(fx: &mut FunctionCx<'_, 'tcx, impl cranelift_module::Backend>, dest_layout: TyLayout<'tcx>) -> CPlace<'tcx> {
    let true_ = fx.bcx.ins().iconst(types::I32, 1);
    fx.bcx.ins().trapnz(true_, TrapCode::User(!0));
    let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
    CPlace::Addr(zero, None, dest_layout)
}
