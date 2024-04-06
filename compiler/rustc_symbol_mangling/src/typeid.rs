/// Type metadata identifiers for LLVM Control Flow Integrity (CFI) and cross-language LLVM CFI
/// support.
///
/// For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
/// see design document in the tracking issue #89653.
use bitflags::bitflags;
use rustc_middle::ty::{Instance, InstanceDef, ReifyReason, Ty, TyCtxt};
use rustc_target::abi::call::FnAbi;
use std::hash::Hasher;
use twox_hash::XxHash64;

bitflags! {
    /// Options for typeid_for_fnabi.
    #[derive(Clone, Copy, Debug)]
    pub struct TypeIdOptions: u32 {
        /// Generalizes pointers for compatibility with Clang
        /// `-fsanitize-cfi-icall-generalize-pointers` option for cross-language LLVM CFI and KCFI
        /// support.
        const GENERALIZE_POINTERS = 1;
        /// Generalizes repr(C) user-defined type for extern function types with the "C" calling
        /// convention (or extern types) for cross-language LLVM CFI and  KCFI support.
        const GENERALIZE_REPR_C = 2;
        /// Normalizes integers for compatibility with Clang
        /// `-fsanitize-cfi-icall-experimental-normalize-integers` option for cross-language LLVM
        /// CFI and  KCFI support.
        const NORMALIZE_INTEGERS = 4;
        /// Do not perform self type erasure for attaching a secondary type id to methods with their
        /// concrete self so they can be used as function pointers.
        ///
        /// (This applies to typeid_for_instance only and should be used to attach a secondary type
        /// id to methods during their declaration/definition so they match the type ids returned by
        /// either typeid_for_instance or typeid_for_fnabi at call sites during code generation for
        /// type membership tests when methods are used as function pointers.)
        const USE_CONCRETE_SELF = 8;
    }
}

mod typeid_itanium_cxx_abi;

/// Returns a type metadata identifier for the specified FnAbi.
pub fn typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> String {
    typeid_itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, options)
}

/// Returns a type metadata identifier for the specified Instance.
pub fn typeid_for_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    options: TypeIdOptions,
) -> String {
    typeid_itanium_cxx_abi::typeid_for_instance(tcx, instance, options)
}

/// Returns a KCFI type metadata identifier for the specified FnAbi.
pub fn kcfi_typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> u32 {
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(typeid_itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, options).as_bytes());
    hash.finish() as u32
}

/// Returns a KCFI type metadata identifier for the specified Instance.
pub fn kcfi_typeid_for_instance<'tcx>(
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
    if matches!(instance.def, InstanceDef::ReifyShim(_, Some(ReifyReason::FnPtr))) {
        options.insert(TypeIdOptions::USE_CONCRETE_SELF);
    }
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(typeid_itanium_cxx_abi::typeid_for_instance(tcx, instance, options).as_bytes());
    hash.finish() as u32
}
