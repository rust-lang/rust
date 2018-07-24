// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Reference: RISC-V ELF psABI specification
// https://github.com/riscv/riscv-elf-psabi-doc

use abi::call::{ArgType, FnType};

fn classify_ret_ty<Ty>(arg: &mut ArgType<Ty>, xlen: u64) {
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

fn classify_arg_ty<Ty>(arg: &mut ArgType<Ty>, xlen: u64) {
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

pub fn compute_abi_info<Ty>(fty: &mut FnType<Ty>, xlen: u64) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(&mut fty.ret, xlen);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(arg, xlen);
    }
}
