//! Classifies types for the pointer-split and type-hash-pointer-split heap partitioning schemes.
//!
//! For more information about LLVM AllocToken and heap partitioning support for the Rust compiler,
//! see design document in the tracking issue #159111.

use rustc_hir::find_attr;
use rustc_middle::ty::{self, IntTy, Ty, TyCtxt, UintTy};

/// Returns whether the given type contains pointers, used to classify types for the pointer-split
/// and type-hash-pointer-split heap partitioning schemes.
///
/// * Recursively classifies the types composed by value in the given type (i.e., elements of
///   arrays, slices, and tuples, fields of structs, enums, and unions, and closure and coroutine
///   captured state).
/// * Classifies references, raw pointers, and function pointers and function items as
///   containing pointers without examining the pointed-to type.
/// * Conservatively classifies types that can not be properly classified, such as trait
///   objects and extern types, as containing pointers.
/// * Classifies isize and usize as containing pointers, consistent with how Clang classifies
///   `uintptr_t` and `intptr_t`.
pub(super) fn contains_pointer<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // Classify isize and usize as containing pointers, consistent with how Clang classifies
        // `uintptr_t` and `intptr_t`, and because pointers may be stored as usize (e.g., via
        // `ptr::expose_provenance` and `ptr::with_exposed_provenance`, and `AtomicUsize`).
        ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => true,

        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str | ty::Never => {
            false
        }

        ty::Array(elem_ty, _) | ty::Slice(elem_ty) | ty::Pat(elem_ty, _) => {
            contains_pointer(tcx, *elem_ty)
        }

        ty::Tuple(tys) => tys.iter().any(|ty| contains_pointer(tcx, ty)),

        ty::Adt(adt_def, args) => {
            // Use user-defined contains-pointer classification, if present (see
            // `#[alloc_token_hint]`).
            if let Some(contains_pointers) = find_attr!(
                tcx,
                adt_def.did(),
                AllocTokenHint { contains_pointers, .. } => contains_pointers
            )
            .copied()
            .flatten()
            {
                return contains_pointers;
            }

            // Classify structs, enums, and unions as containing pointers if any field type of any
            // variant contains pointers.
            adt_def
                .all_fields()
                .any(|field| contains_pointer(tcx, field.ty(tcx, args).skip_norm_wip()))
        }

        ty::Closure(_, args) => {
            args.as_closure().upvar_tys().iter().any(|ty| contains_pointer(tcx, ty))
        }
        ty::CoroutineClosure(_, args) => {
            args.as_coroutine_closure().upvar_tys().iter().any(|ty| contains_pointer(tcx, ty))
        }
        ty::Coroutine(_, args) => {
            args.as_coroutine().upvar_tys().iter().any(|ty| contains_pointer(tcx, ty))
        }

        // Classify references, raw pointers, and function pointers and function items as
        // containing pointers without examining the pointed-to type.
        ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) | ty::FnDef(..) => true,

        // Conservatively classify trait objects (unknown concrete type), extern types (opaque), and
        // all other types that can not be properly classified as containing pointers, because
        // misclassifying a type not containing pointers into the partition containing pointers only
        // loses part of the separation benefit for that type, while misclassifying a type
        // containing pointers into the partition not containing pointers breaks the guarantee of
        // the partition not containing pointers.
        ty::Foreign(_)
        | ty::Dynamic(..)
        | ty::CoroutineWitness(..)
        | ty::UnsafeBinder(_)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(_)
        | ty::Infer(_)
        | ty::Error(_) => true,
    }
}
