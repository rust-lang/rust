//! Encodes type names for the pointer-split and type-hash-pointer-split heap partitioning schemes.
//!
//! For more information about LLVM AllocToken and heap partitioning support for the Rust compiler,
//! see design document in the tracking issue #159111.

use rustc_hir::find_attr;
use rustc_middle::ty::{self, Ty, TyCtxt};

/// Encodes a ty:Ty in a type name uniquely identifying the given type for the pointer-split and
/// type-hash-pointer-split heap partitioning schemes.
pub(super) fn encode_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> String {
    if let ty::Adt(adt_def, _) = ty.kind() {
        // Use user-defined type name encoding, if present (see `#[alloc_token_hint]`).
        if let Some(type_name) = find_attr!(
            tcx,
            adt_def.did(),
            AllocTokenHint { type_name, .. } => type_name
        )
        .copied()
        .flatten()
        {
            return type_name.to_string();
        }

        // For cross-language LLVM AllocToken and heap partitioning support, the type name encoding
        // must be compatible for types used at the FFI boundary. For instance:
        //
        //     struct Foo { void *next; };
        //
        // Is encoded as "Foo" (i.e., Clang emits the canonical, fully qualified C and C++ type name
        // as the type name, which for repr(C) types is the type name without a Rust crate name and
        // path names prefix). So, encode any repr(C) user-defined type as its plain, unscoped name.
        if adt_def.repr().c() {
            return tcx.item_name(adt_def.did()).to_string();
        }
    }

    // Use the Rust v0 symbol mangling type encoding for all remaining Rust types.
    rustc_symbol_mangling::typeid_for_ty(tcx, ty)
}
