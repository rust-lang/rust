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
//! > - Doxygen Documentation of `clang::DefaultABIInfo`
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

use rustc_abi::{CanonAbi, TyAbiInterface};

use crate::callconv::{ArgAbi, ComputeCallConv};

fn classify_ret_ty<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() {
        ret.make_indirect();
    }
}

fn classify_arg_ty<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if arg.layout.is_aggregate() {
        arg.make_indirect();
    }
}

pub(crate) struct AvrCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for AvrCallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn compute_ret_abi(&mut self, _cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if !ret.is_ignore() {
            classify_ret_ty(ret);
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            return;
        }

        classify_arg_ty(cx, arg);
    }
}
