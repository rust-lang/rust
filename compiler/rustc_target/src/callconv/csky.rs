// Reference: CSKY ABI Manual
// https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1695027452256/T-HEAD_800_Series_ABI_Standards_Manual.pdf
//
// Reference: Clang CSKY lowering code
// https://github.com/llvm/llvm-project/blob/4a074f32a6914f2a8d7215d78758c24942dddc3d/clang/lib/CodeGen/Targets/CSKY.cpp#L76-L162

use crate::callconv::{ArgAbi, FnAbi, Reg, Uniform};

fn classify_ret<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    // For return type, aggregate which <= 2*XLen will be returned in registers.
    // Otherwise, aggregate will be returned indirectly.
    if arg.layout.is_aggregate() {
        let total = arg.layout.size;
        if total.bits() > 64 {
            arg.make_indirect();
        } else if total.bits() > 32 {
            arg.cast_to(Uniform::new(Reg::i32(), total));
        } else {
            arg.cast_to(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }
}

fn classify_arg<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    // For argument type, the first 4*XLen parts of aggregate will be passed
    // in registers, and the rest will be passed in stack.
    // So we can coerce to integers directly and let backend handle it correctly.
    if arg.layout.is_aggregate() {
        let total = arg.layout.size;
        if total.bits() > 32 {
            arg.cast_to(Uniform::new(Reg::i32(), total));
        } else {
            arg.cast_to(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(arg);
    }
}
