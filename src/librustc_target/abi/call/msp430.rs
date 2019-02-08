// Reference: MSP430 Embedded Application Binary Interface
// http://www.ti.com/lit/an/slaa534/slaa534.pdf

use crate::abi::call::{ArgType, FnType};

// 3.5 Structures or Unions Passed and Returned by Reference
//
// "Structures (including classes) and unions larger than 32 bits are passed and
// returned by reference. To pass a structure or union by reference, the caller
// places its address in the appropriate location: either in a register or on
// the stack, according to its position in the argument list. (..)"
fn classify_ret_ty<Ty>(ret: &mut ArgType<'_, Ty>) {
    if ret.layout.is_aggregate() && ret.layout.size.bits() > 32 {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(16);
    }
}

fn classify_arg_ty<Ty>(arg: &mut ArgType<'_, Ty>) {
    if arg.layout.is_aggregate() && arg.layout.size.bits() > 32 {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(16);
    }
}

pub fn compute_abi_info<Ty>(fty: &mut FnType<'_, Ty>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(&mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(arg);
    }
}
