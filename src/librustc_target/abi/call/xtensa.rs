// reference: https://github.com/espressif/clang-xtensa/commit/6fb488d2553f06029e6611cf81c6efbd45b56e47#diff-aa74ae1e1ab6b7149789237edb78e688R8450


use crate::abi::call::{ArgType, FnType, Reg, Uniform};

const NUM_ARG_GPR: u64 = 6;
const MAX_ARG_IN_REGS_SIZE: u64  = 4 * 32;
// const MAX_ARG_DIRECT_SIZE : u64 = MAX_ARG_IN_REGS_SIZE;
const MAX_RET_IN_REGS_SIZE: u64  = 2 * 32;

fn classify_ret_ty<Ty>(arg: &mut ArgType<'_, Ty>, xlen: u64) {
    // The rules for return and argument types are the same, so defer to
    // classifyArgumentType.
    classify_arg_ty(arg, xlen, &mut 2); // two as max return size
}


fn classify_arg_ty<Ty>(arg: &mut ArgType<'_, Ty>, xlen: u64, remaining_gpr: &mut u64) {
    // Determine the number of GPRs needed to pass the current argument
    // according to the ABI. 2*XLen-aligned varargs are passed in "aligned"
    // register pairs, so may consume 3 registers.
    
    let mut stack_required = false;
    let mut required_gpr = 1u64; // at least one to start
    let arg_size = arg.layout.size.bits();
    // let alignment = arg.layout.details.align.abi.bits();


    if  arg_size > xlen && arg_size <= MAX_ARG_IN_REGS_SIZE {
        required_gpr = arg_size + (xlen - 1) / xlen; 
    }

    if required_gpr > *remaining_gpr {
        stack_required = true;
    required_gpr = *remaining_gpr; }


    *remaining_gpr -= required_gpr;

    // if  a value can fit in a reg and the
    // stack is not required, extend
    if !arg.layout.is_aggregate() {
        if arg_size < xlen && !stack_required {
            arg.extend_integer_width_to(xlen);
        }
        return;
    }


    if arg_size as u64 <= MAX_ARG_IN_REGS_SIZE {
        let align = arg.layout.align.abi.bytes();
        let total = arg.layout.size;
        arg.cast_to(Uniform {
            unit: if align <= 4 { Reg::i32() } else { Reg::i64() },
            total
        });
        return;
    }
        
    // if we get here the stack is required
    // assert!(stack_required);
    arg.make_indirect();
}

pub fn compute_abi_info<Ty>(fty: &mut FnType<'_, Ty>, xlen: u64) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(&mut fty.ret, xlen);
    }

    let return_indirect = fty.ret.layout.size.bits() > MAX_RET_IN_REGS_SIZE ||
                            fty.ret.is_indirect();

    let mut remaining_gpr = if return_indirect {
        NUM_ARG_GPR - 1
    } else {
        NUM_ARG_GPR
    };

    for arg in &mut fty.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(arg, xlen, &mut remaining_gpr);
    }
}
