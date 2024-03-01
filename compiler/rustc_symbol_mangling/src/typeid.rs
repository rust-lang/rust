/// Type metadata identifiers for LLVM Control Flow Integrity (CFI) and cross-language LLVM CFI
/// support.
///
/// For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
/// see design document in the tracking issue #89653.
use bitflags::bitflags;
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use rustc_target::abi::call::FnAbi;
use std::hash::Hasher;
use twox_hash::XxHash64;

bitflags! {
    /// Options for typeid_for_fnabi.
    #[derive(Clone, Copy, Debug)]
    pub struct TypeIdOptions: u32 {
        const GENERALIZE_POINTERS = 1;
        const GENERALIZE_REPR_C = 2;
        const NORMALIZE_INTEGERS = 4;
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
    instance: &Instance<'tcx>,
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
    instance: &Instance<'tcx>,
    options: TypeIdOptions,
) -> u32 {
    // A KCFI type metadata identifier is a 32-bit constant produced by taking the lower half of the
    // xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(typeid_itanium_cxx_abi::typeid_for_instance(tcx, instance, options).as_bytes());
    hash.finish() as u32
}
