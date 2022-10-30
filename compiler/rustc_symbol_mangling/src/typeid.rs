// For more information about type metadata and type metadata identifiers for cross-language LLVM
// CFI support, see Type metadata in the design document in the tracking issue #89653.

use rustc_middle::ty::{FnSig, Ty, TyCtxt};
use rustc_target::abi::call::FnAbi;

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
