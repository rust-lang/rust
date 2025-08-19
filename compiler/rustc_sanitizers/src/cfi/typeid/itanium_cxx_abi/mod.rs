//! Type metadata identifiers (using Itanium C++ ABI mangling for encoding) for LLVM Control Flow
//! Integrity (CFI) and cross-language LLVM CFI support.
//!
//! For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
//! see design document in the tracking issue #89653.

use rustc_abi::CanonAbi;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::bug;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_target::callconv::{FnAbi, PassMode};
use tracing::instrument;

mod encode;
mod transform;
use crate::cfi::typeid::TypeIdOptions;
use crate::cfi::typeid::itanium_cxx_abi::encode::{DictKey, EncodeTyOptions, encode_ty};
use crate::cfi::typeid::itanium_cxx_abi::transform::{
    TransformTy, TransformTyOptions, transform_instance,
};

/// Returns a type metadata identifier for the specified FnAbi using the Itanium C++ ABI with vendor
/// extended type qualifiers and types for Rust types that are not used at the FFI boundary.
#[instrument(level = "trace", skip(tcx))]
pub fn typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> String {
    // A name is mangled by prefixing "_Z" to an encoding of its name, and in the case of functions
    // its type.
    let mut typeid = String::from("_Z");

    // Clang uses the Itanium C++ ABI's virtual tables and RTTI typeinfo structure name as type
    // metadata identifiers for function pointers. The typeinfo name encoding is a two-character
    // code (i.e., 'TS') prefixed to the type encoding for the function.
    typeid.push_str("TS");

    // Function types are delimited by an "F..E" pair
    typeid.push('F');

    // A dictionary of substitution candidates used for compression (see
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-compression).
    let mut dict: FxHashMap<DictKey<'tcx>, usize> = FxHashMap::default();

    let mut encode_ty_options = EncodeTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("typeid_for_fnabi: invalid option(s) `{:?}`", options.bits()));
    match fn_abi.conv {
        CanonAbi::C => {
            encode_ty_options.insert(EncodeTyOptions::GENERALIZE_REPR_C);
        }
        _ => {
            encode_ty_options.remove(EncodeTyOptions::GENERALIZE_REPR_C);
        }
    }

    // Encode the return type
    let transform_ty_options = TransformTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("typeid_for_fnabi: invalid option(s) `{:?}`", options.bits()));
    let mut type_folder = TransformTy::new(tcx, transform_ty_options);
    let ty = fn_abi.ret.layout.ty.fold_with(&mut type_folder);
    typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));

    // Encode the parameter types

    // We erase ZSTs as we go if the argument is skipped. This is an implementation detail of how
    // MIR is currently treated by rustc, and subject to change in the future. Specifically, MIR
    // interpretation today will allow skipped arguments to simply not be passed at a call-site.
    if !fn_abi.c_variadic {
        let mut pushed_arg = false;
        for arg in fn_abi.args.iter().filter(|arg| arg.mode != PassMode::Ignore) {
            pushed_arg = true;
            let ty = arg.layout.ty.fold_with(&mut type_folder);
            typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));
        }
        if !pushed_arg {
            // Empty parameter lists, whether declared as () or conventionally as (void), are
            // encoded with a void parameter specifier "v".
            typeid.push('v');
        }
    } else {
        for n in 0..fn_abi.fixed_count as usize {
            if fn_abi.args[n].mode == PassMode::Ignore {
                continue;
            }
            let ty = fn_abi.args[n].layout.ty.fold_with(&mut type_folder);
            typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));
        }

        typeid.push('z');
    }

    // Close the "F..E" pair
    typeid.push('E');

    // Add encoding suffixes
    if options.contains(EncodeTyOptions::NORMALIZE_INTEGERS) {
        typeid.push_str(".normalized");
    }

    if options.contains(EncodeTyOptions::GENERALIZE_POINTERS) {
        typeid.push_str(".generalized");
    }

    typeid
}

/// Returns a type metadata identifier for the specified Instance using the Itanium C++ ABI with
/// vendor extended type qualifiers and types for Rust types that are not used at the FFI boundary.
#[instrument(level = "trace", skip(tcx))]
pub fn typeid_for_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    options: TypeIdOptions,
) -> String {
    assert!(!instance.has_non_region_param(), "{instance:#?} must be fully monomorphic");
    let transform_ty_options = TransformTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("typeid_for_instance: invalid option(s) `{:?}`", options.bits()));
    let instance = transform_instance(tcx, instance, transform_ty_options);
    let fn_abi = tcx
        .fn_abi_of_instance(
            ty::TypingEnv::fully_monomorphized().as_query_input((instance, ty::List::empty())),
        )
        .unwrap_or_else(|error| {
            bug!("typeid_for_instance: couldn't get fn_abi of instance {instance:?}: {error:?}")
        });
    typeid_for_fnabi(tcx, fn_abi, options)
}
