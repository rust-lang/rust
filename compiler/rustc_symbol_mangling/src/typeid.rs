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
        /// Generalize the instance by erasing the concrete `Self` type where possible.
        /// Only has an effect on `{kcfi_,}typeid_for_instance`.
        const ERASE_SELF_TYPE = 8;
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
    // If we receive a `ReifyShim` intended to produce a function pointer, we need to remain
    // concrete - abstraction is for vtables.
    if matches!(instance.def, InstanceDef::ReifyShim(_, Some(ReifyReason::FnPtr))) {
        options.remove(TypeIdOptions::ERASE_SELF_TYPE);
    }
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(typeid_itanium_cxx_abi::typeid_for_instance(tcx, instance, options).as_bytes());
    hash.finish() as u32
}
