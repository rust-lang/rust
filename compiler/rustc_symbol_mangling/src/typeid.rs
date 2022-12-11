// For more information about type metadata and type metadata identifiers for cross-language LLVM
// CFI support, see Type metadata in the design document in the tracking issue #89653.

use rustc_middle::ty::{FnSig, Ty, TyCtxt};
use rustc_target::abi::call::FnAbi;
use std::hash::Hasher;
use twox_hash::XxHash64;

mod typeid_itanium_cxx_abi;
use typeid_itanium_cxx_abi::TypeIdOptions;

/// Returns a type metadata identifier for the specified FnAbi.
pub fn typeid_for_fnabi<'tcx>(tcx: TyCtxt<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> String {
    typeid_itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, TypeIdOptions::NO_OPTIONS)
}

/// Returns a type metadata identifier for the specified FnSig.
pub fn typeid_for_fnsig<'tcx>(tcx: TyCtxt<'tcx>, fn_sig: &FnSig<'tcx>) -> String {
    typeid_itanium_cxx_abi::typeid_for_fnsig(tcx, fn_sig, TypeIdOptions::NO_OPTIONS)
}

/// Returns an LLVM KCFI type metadata identifier for the specified FnAbi.
pub fn kcfi_typeid_for_fnabi<'tcx>(tcx: TyCtxt<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> u32 {
    // An LLVM KCFI type metadata identifier is a 32-bit constant produced by taking the lower half
    // of the xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(
        typeid_itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, TypeIdOptions::NO_OPTIONS).as_bytes(),
    );
    hash.finish() as u32
}

/// Returns an LLVM KCFI type metadata identifier for the specified FnSig.
pub fn kcfi_typeid_for_fnsig<'tcx>(tcx: TyCtxt<'tcx>, fn_sig: &FnSig<'tcx>) -> u32 {
    // An LLVM KCFI type metadata identifier is a 32-bit constant produced by taking the lower half
    // of the xxHash64 of the type metadata identifier. (See llvm/llvm-project@cff5bef.)
    let mut hash: XxHash64 = Default::default();
    hash.write(
        typeid_itanium_cxx_abi::typeid_for_fnsig(tcx, fn_sig, TypeIdOptions::NO_OPTIONS).as_bytes(),
    );
    hash.finish() as u32
}
