// reference: https://github.com/MabezDev/llvm-project/blob/xtensa_release_9.0.1_with_rust_patches-31-05-2020-cherry-pick/clang/lib/CodeGen/TargetInfo.cpp#L9668-L9767

use crate::abi::call::{ArgAbi, FnAbi, Reg, Uniform};
use crate::abi::{Abi, Size};

const NUM_ARG_GPRS: u64 = 6;
const MAX_ARG_IN_REGS_SIZE: u64 = 4 * 32;
const MAX_RET_IN_REGS_SIZE: u64 = 2 * 32;

fn classify_ret_ty<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64) {
    if arg.is_ignore() {
        return;
    }

    // The rules for return and argument types are the same,
    // so defer to `classify_arg_ty`.
    let mut arg_gprs_left = 2;
    let fixed = true;
    classify_arg_ty(arg, xlen, fixed, &mut arg_gprs_left);
}

fn classify_arg_ty<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64, fixed: bool, arg_gprs_left: &mut u64) {
    assert!(*arg_gprs_left <= NUM_ARG_GPRS, "Arg GPR tracking underflow");

    // Ignore empty structs/unions.
    if arg.layout.is_zst() {
        return;
    }

    let size = arg.layout.size.bits();
    let needed_align = arg.layout.align.abi.bits();
    let mut must_use_stack = false;

    // Determine the number of GPRs needed to pass the current argument
    // according to the ABI. 2*XLen-aligned varargs are passed in "aligned"
    // register pairs, so may consume 3 registers.
    let mut needed_arg_gprs = 1u64;

    if !fixed && needed_align == 2 * xlen {
        needed_arg_gprs = 2 + (*arg_gprs_left % 2);
    } else if size > xlen && size <= MAX_ARG_IN_REGS_SIZE {
        needed_arg_gprs = (size + xlen - 1) / xlen;
    }

    if needed_arg_gprs > *arg_gprs_left {
        must_use_stack = true;
        needed_arg_gprs = *arg_gprs_left;
    }
    *arg_gprs_left -= needed_arg_gprs;

    if !arg.layout.is_aggregate() && !matches!(arg.layout.abi, Abi::Vector { .. }) {
        // All integral types are promoted to `xlen`
        // width, unless passed on the stack.
        if size < xlen && !must_use_stack {
            arg.extend_integer_width_to(xlen);
            return;
        }

        return;
    }

    // Aggregates which are <= 4 * 32 will be passed in
    // registers if possible, so coerce to integers.
    if size as u64 <= MAX_ARG_IN_REGS_SIZE {
        let alignment = arg.layout.align.abi.bits();

        // Use a single `xlen` int if possible, 2 * `xlen` if 2 * `xlen` alignment
        // is required, and a 2-element `xlen` array if only `xlen` alignment is
        // required.
        if size <= xlen {
            arg.cast_to(Reg::i32());
            return;
        } else if alignment == 2 * xlen {
            arg.cast_to(Reg::i64());
            return;
        } else {
            let total = Size::from_bits(((size + xlen - 1) / xlen) * xlen);
            arg.cast_to(Uniform { unit: Reg::i32(), total });
            return;
        }
    }

    arg.make_indirect();
}

pub fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>, xlen: u64) {
    classify_ret_ty(&mut fn_abi.ret, xlen);

    let is_ret_indirect =
        fn_abi.ret.is_indirect() || fn_abi.ret.layout.size.bits() > MAX_RET_IN_REGS_SIZE;

    let mut arg_gprs_left = if is_ret_indirect { NUM_ARG_GPRS - 1 } else { NUM_ARG_GPRS };

    for arg in &mut fn_abi.args {
        let fixed = true;
        classify_arg_ty(arg, xlen, fixed, &mut arg_gprs_left);
    }
}
