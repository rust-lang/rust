// reference: https://github.com/espressif/clang-xtensa/commit/6fb488d2553f06029e6611cf81c6efbd45b56e47#diff-aa74ae1e1ab6b7149789237edb78e688R8450


use crate::abi::call::{ArgAbi, FnAbi, Reg, Uniform};

const NUM_ARG_GPR: u64 = 6;
const MAX_ARG_IN_REGS_SIZE: u64  = 4 * 32;
// const MAX_ARG_DIRECT_SIZE: u64 = MAX_ARG_IN_REGS_SIZE;
const MAX_RET_IN_REGS_SIZE: u64  = 2 * 32;

fn classify_ret_ty<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64) {
    // The rules for return and argument types are the same, so defer to
    // classifyArgumentType.
    classify_arg_ty(arg, xlen, &mut 2); // two as max return size
}


fn classify_arg_ty<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64, remaining_gpr: &mut u64) {
    // Determine the number of GPRs needed to pass the current argument
    // according to the ABI. 2*XLen-aligned varargs are passed in "aligned"
    // register pairs, so may consume 3 registers.
    
    let arg_size = arg.layout.size;
    if arg_size.bits() > MAX_ARG_IN_REGS_SIZE {
        arg.make_indirect();
        return;
    }

    let alignment = arg.layout.details.align.abi;
    let mut required_gpr = 1u64; // at least one per arg

    if alignment.bits() == 2 * xlen {
        required_gpr = 2 + (*remaining_gpr % 2);
    } else if  arg_size.bits() > xlen && arg_size.bits() <= MAX_ARG_IN_REGS_SIZE {
        required_gpr = (arg_size.bits() + (xlen - 1)) / xlen;
    }

    let mut stack_required = false;
    if required_gpr > *remaining_gpr {
        stack_required = true;
        required_gpr = *remaining_gpr;
    }
    *remaining_gpr -= required_gpr;

    // if  a value can fit in a reg and the
    // stack is not required, extend
    if !arg.layout.is_aggregate() { // non-aggregate types
        if arg_size.bits() < xlen && !stack_required {
            arg.extend_integer_width_to(xlen);
        }
    } else if arg_size.bits() as u64 <= MAX_ARG_IN_REGS_SIZE { // aggregate types
        // Aggregates which are <= 4*32 will be passed in registers if possible,
        // so coerce to integers.
        
        // Use a single XLen int if possible, 2*XLen if 2*XLen alignment is
        // required, and a 2-element XLen array if only XLen alignment is
        // required.
        // if alignment == 2 * xlen {
        //     arg.extend_integer_width_to(xlen * 2);
        // } else {
        //     arg.extend_integer_width_to(arg_size + (xlen - 1) / xlen);
        // }
        if alignment.bits() == 2 * xlen {
            arg.cast_to(Uniform {
                unit: Reg::i64(),
                total: arg_size
            });
        } else {
            //TODO array type - this should be a homogenous array type
            // arg.extend_integer_width_to(arg_size + (xlen - 1) / xlen);
        }
        
    } else {
        // if we get here the stack is required
        assert!(stack_required);
        arg.make_indirect();
    }


   // if arg_size as u64 <= MAX_ARG_IN_REGS_SIZE {
   //     let align = arg.layout.align.abi.bytes();
   //     let total = arg.layout.size;
   //     arg.cast_to(Uniform {
   //         unit: if align <= 4 { Reg::i32() } else { Reg::i64() },
   //         total
   //     });
   //     return;
   // }
        
    
}

pub fn compute_abi_info<Ty>(fabi: &mut FnAbi<'_, Ty>, xlen: u64) {
    if !fabi.ret.is_ignore() {
        classify_ret_ty(&mut fabi.ret, xlen);
    }

    let return_indirect = fabi.ret.layout.size.bits() > MAX_RET_IN_REGS_SIZE ||
                            fabi.ret.is_indirect();

    let mut remaining_gpr = if return_indirect {
        NUM_ARG_GPR - 1
    } else {
        NUM_ARG_GPR
    };

    for arg in &mut fabi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(arg, xlen, &mut remaining_gpr);
    }
}
