// Reference: PTX Writer's Guide to Interoperability
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability

use crate::abi::call::{ArgType, FnType};

fn classify_ret_ty<Ty>(ret: &mut ArgType<'_, Ty>) {
    if ret.layout.is_aggregate() && ret.layout.size.bits() > 64 {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(64);
    }
}

fn classify_arg_ty<Ty>(arg: &mut ArgType<'_, Ty>) {
    if arg.layout.is_aggregate() && arg.layout.size.bits() > 64 {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(64);
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
