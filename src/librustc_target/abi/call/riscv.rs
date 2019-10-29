// Reference: RISC-V ELF psABI specification
// https://github.com/riscv/riscv-elf-psabi-doc

use crate::abi::call::{ArgAbi, FnAbi};

fn classify_ret<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64) {
    // "Scalars wider than 2✕XLEN are passed by reference and are replaced in
    // the argument list with the address."
    // "Aggregates larger than 2✕XLEN bits are passed by reference and are
    // replaced in the argument list with the address, as are C++ aggregates
    // with nontrivial copy constructors, destructors, or vtables."
    if arg.layout.size.bits() > 2 * xlen {
        arg.make_indirect();
    }

    // "When passed in registers, scalars narrower than XLEN bits are widened
    // according to the sign of their type up to 32 bits, then sign-extended to
    // XLEN bits."
    arg.extend_integer_width_to(xlen); // this method only affects integer scalars
}

fn classify_arg<Ty>(arg: &mut ArgAbi<'_, Ty>, xlen: u64) {
    // "Scalars wider than 2✕XLEN are passed by reference and are replaced in
    // the argument list with the address."
    // "Aggregates larger than 2✕XLEN bits are passed by reference and are
    // replaced in the argument list with the address, as are C++ aggregates
    // with nontrivial copy constructors, destructors, or vtables."
    if arg.layout.size.bits() > 2 * xlen {
        arg.make_indirect();
    }

    // "When passed in registers, scalars narrower than XLEN bits are widened
    // according to the sign of their type up to 32 bits, then sign-extended to
    // XLEN bits."
    arg.extend_integer_width_to(xlen); // this method only affects integer scalars
}

pub fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>, xlen: u64) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret, xlen);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(arg, xlen);
    }
}
