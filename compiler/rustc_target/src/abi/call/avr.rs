//! LLVM-frontend specific AVR calling convention implementation.
//!
//! # Current calling convention ABI
//!
//! Inherited from Clang's `clang::DefaultABIInfo` implementation - self described
//! as
//!
//! > the default implementation for ABI specific details. This implementation
//! > provides information which results in
//! > self-consistent and sensible LLVM IR generation, but does not
//! > conform to any particular ABI.
//! >
//! > - Doxygen Doxumentation of `clang::DefaultABIInfo`
//!
//! This calling convention may not match AVR-GCC in all cases.
//!
//! In the future, an AVR-GCC compatible argument classification ABI should be
//! adopted in both Rust and Clang.
//!
//! *NOTE*: Currently, this module implements the same calling convention
//! that clang with AVR currently does - the default, simple, unspecialized
//! ABI implementation available to all targets. This ABI is not
//! binary-compatible with AVR-GCC. Once LLVM [PR46140](https://bugs.llvm.org/show_bug.cgi?id=46140)
//! is completed, this module should be updated to match so that both Clang
//! and Rust emit code to the same AVR-GCC compatible ABI.
//!
//! In particular, both Clang and Rust may not have the same semantics
//! when promoting arguments to indirect references as AVR-GCC. It is important
//! to note that the core AVR ABI implementation within LLVM itself is ABI
//! compatible with AVR-GCC - Rust and AVR-GCC only differ in the small amount
//! of compiler frontend specific calling convention logic implemented here.

use crate::abi::call::{ArgAbi, FnAbi};

fn classify_ret_ty<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() {
        ret.make_indirect();
    }
}

fn classify_arg_ty<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    if arg.layout.is_aggregate() {
        arg.make_indirect();
    }
}

pub fn compute_abi_info<Ty>(fty: &mut FnAbi<'_, Ty>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(&mut fty.ret);
    }

    for arg in fty.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }

        classify_arg_ty(arg);
    }
}
