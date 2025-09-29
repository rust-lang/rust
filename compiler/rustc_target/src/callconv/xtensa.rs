//! The Xtensa ABI implementation
//!
//! This ABI implementation is based on the following sources:
//!
//! Section 8.1.4 & 8.1.5 of the Xtensa ISA reference manual, as well as snippets from
//! Section 2.3 from the Xtensa programmers guide.

use rustc_abi::{BackendRepr, HasDataLayout, Size, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, Uniform};
use crate::spec::HasTargetSpec;

const NUM_ARG_GPRS: u64 = 6;
const NUM_RET_GPRS: u64 = 4;
const MAX_ARG_IN_REGS_SIZE: u64 = NUM_ARG_GPRS * 32;
const MAX_RET_IN_REGS_SIZE: u64 = NUM_RET_GPRS * 32;

fn classify_ret_ty<'a, Ty, C>(arg: &mut ArgAbi<'_, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.is_ignore() {
        return;
    }

    // The rules for return and argument types are the same,
    // so defer to `classify_arg_ty`.
    let mut arg_gprs_left = NUM_RET_GPRS;
    classify_arg_ty(arg, &mut arg_gprs_left, MAX_RET_IN_REGS_SIZE);
    // Ret args cannot be passed via stack, we lower to indirect and let the backend handle the invisible reference
    match arg.mode {
        super::PassMode::Indirect { attrs: _, meta_attrs: _, ref mut on_stack } => {
            *on_stack = false;
        }
        _ => {}
    }
}

fn classify_arg_ty<'a, Ty, C>(arg: &mut ArgAbi<'_, Ty>, arg_gprs_left: &mut u64, max_size: u64)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    assert!(*arg_gprs_left <= NUM_ARG_GPRS, "Arg GPR tracking underflow");

    // Ignore empty structs/unions.
    if arg.layout.is_zst() {
        return;
    }

    let size = arg.layout.size.bits();
    let needed_align = arg.layout.align.bits();
    let mut must_use_stack = false;

    // Determine the number of GPRs needed to pass the current argument
    // according to the ABI. 2*XLen-aligned varargs are passed in "aligned"
    // register pairs, so may consume 3 registers.
    let mut needed_arg_gprs = size.div_ceil(32);
    if needed_align == 64 {
        needed_arg_gprs += *arg_gprs_left % 2;
    }

    if needed_arg_gprs > *arg_gprs_left
        || needed_align > 128
        || (*arg_gprs_left < (max_size / 32) && needed_align == 128)
    {
        must_use_stack = true;
        needed_arg_gprs = *arg_gprs_left;
    }
    *arg_gprs_left -= needed_arg_gprs;

    if must_use_stack {
        arg.pass_by_stack_offset(None);
    } else if is_xtensa_aggregate(arg) {
        // Aggregates which are <= max_size will be passed in
        // registers if possible, so coerce to integers.

        // Use a single `xlen` int if possible, 2 * `xlen` if 2 * `xlen` alignment
        // is required, and a 2-element `xlen` array if only `xlen` alignment is
        // required.
        if size <= 32 {
            arg.cast_to(Reg::i32());
        } else {
            let reg = if needed_align == 2 * 32 { Reg::i64() } else { Reg::i32() };
            let total = Size::from_bits(((size + 32 - 1) / 32) * 32);
            arg.cast_to(Uniform::new(reg, total));
        }
    } else {
        // All integral types are promoted to `xlen`
        // width.
        //
        // We let the LLVM backend handle integral types >= xlen.
        if size < 32 {
            arg.extend_integer_width_to(32);
        }
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(_cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret_ty(&mut fn_abi.ret);
    }

    let mut arg_gprs_left = NUM_ARG_GPRS;

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(arg, &mut arg_gprs_left, MAX_ARG_IN_REGS_SIZE);
    }
}

fn is_xtensa_aggregate<'a, Ty>(arg: &ArgAbi<'a, Ty>) -> bool {
    match arg.layout.backend_repr {
        BackendRepr::SimdVector { .. } => true,
        _ => arg.layout.is_aggregate(),
    }
}
