//! Type metadata identifiers for LLVM Kernel Control Flow Integrity (KCFI) and cross-language LLVM
//! KCFI support for the Rust compiler.
//!
//! For more information about LLVM KCFI and cross-language LLVM KCFI support for the Rust compiler,
//! see the tracking issue #123479.

use std::hash::Hasher;

use rustc_middle::ty::{Instance, InstanceKind, ReifyReason, Ty, TyCtxt};
use rustc_target::callconv::FnAbi;
use twox_hash::XxHash64;

pub use crate::cfi::typeid::{TypeIdOptions, itanium_cxx_abi};

/// Returns a KCFI type metadata identifier for the specified FnAbi.
pub fn typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> u32 {
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, options).as_bytes());
    hash.finish() as u32
}

/// Returns a KCFI type metadata identifier for the specified Instance.
pub fn typeid_for_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    mut options: TypeIdOptions,
) -> u32 {
    // KCFI support for Rust shares most of its implementation with the CFI support, with some key
    // differences:
    //
    // 1. KCFI performs type tests differently and are implemented as different LLVM passes than CFI
    //    to not require LTO.
    // 2. KCFI has the limitation that a function or method may have one type id assigned only.
    //
    // Because of the limitation listed above (2), the current KCFI implementation (not CFI) does
    // reifying of types (i.e., adds shims/trampolines for indirect calls in these cases) for:
    //
    // * Supporting casting between function items, closures, and Fn trait objects.
    // * Supporting methods being cast as function pointers.
    //
    // This was implemented for KCFI support in #123106 and #123052 (which introduced the
    // ReifyReason). The tracking issue for KCFI support for Rust is #123479.
    if matches!(instance.def, InstanceKind::ReifyShim(_, Some(ReifyReason::FnPtr))) {
        options.insert(TypeIdOptions::USE_CONCRETE_SELF);
    }
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(itanium_cxx_abi::typeid_for_instance(tcx, instance, options).as_bytes());
    hash.finish() as u32
}
