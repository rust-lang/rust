//! Type metadata identifiers for LLVM Control Flow Integrity (CFI) and cross-language LLVM CFI
//! support for the Rust compiler.
//!
//! For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
//! see design document in the tracking issue #89653.

use bitflags::bitflags;
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use rustc_target::callconv::FnAbi;

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

pub mod itanium_cxx_abi;

/// Returns a type metadata identifier for the specified FnAbi.
pub fn typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> String {
    itanium_cxx_abi::typeid_for_fnabi(tcx, fn_abi, options)
}

/// Returns a type metadata identifier for the specified Instance.
pub fn typeid_for_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    options: TypeIdOptions,
) -> String {
    itanium_cxx_abi::typeid_for_instance(tcx, instance, options)
}
