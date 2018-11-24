use cranelift::prelude::*;

use cranelift::codegen::ir::TrapCode;

/// Trap code: user0
pub fn trap_panic(bcx: &mut FunctionBuilder) {
    bcx.ins().trap(TrapCode::User(0));
}

/// Trap code: user65535
pub fn trap_unreachable(bcx: &mut FunctionBuilder) {
    bcx.ins().trap(TrapCode::User(!0));
}
