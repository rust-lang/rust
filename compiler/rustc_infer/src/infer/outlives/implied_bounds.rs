//! Extraction of implied bounds from nested references.
//!
//! This module provides utilities for extracting outlives constraints that are
//! implied by the structure of types, particularly nested references.
//!
//! For example, the type `&'a &'b T` implies that `'b: 'a`, because the outer
//! reference with lifetime `'a` must not outlive the data it points to, which
//! has lifetime `'b`.
//!
//! This is relevant for issue #25860, where the combination of variance and
//! implied bounds on nested references can create soundness holes in HRTB
//! function pointer coercions.

use rustc_middle::ty::{self, Ty, TyCtxt};

// Note: Allocation-free helper below is used for fast path decisions.

/// Returns true if the type contains a nested reference structure that implies
/// an outlives relationship (e.g., `&'a &'b T` implies `'b: 'a`). This helper
/// is non-allocating and short-circuits on the first match.
pub fn has_nested_reference_implied_bounds<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    fn walk<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Ref(_, inner_ty, _) => {
                match inner_ty.kind() {
                    // Direct nested reference: &'a &'b T
                    ty::Ref(..) => true,
                    // Recurse into inner type for tuples/ADTs possibly nested within
                    _ => walk(tcx, *inner_ty),
                }
            }
            ty::Tuple(tys) => tys.iter().any(|t| walk(tcx, t)),
            ty::Adt(_, args) => args.iter().any(|arg| match arg.kind() {
                ty::GenericArgKind::Type(t) => walk(tcx, t),
                _ => false,
            }),
            _ => false,
        }
    }

    walk(tcx, ty)
}

/// Returns true if there exists a nested reference `&'a &'b T` within `ty`
/// such that the outer and inner regions are distinct (`'a != 'b`). This
/// helps detect cases where implied outlives like `'b: 'a` exist.
pub fn has_nested_reference_with_distinct_regions<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    fn walk<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Ref(r_outer, inner_ty, _) => {
                match inner_ty.kind() {
                    ty::Ref(r_inner, nested_ty, _) => {
                        if r_outer != r_inner {
                            return true;
                        }
                        // Keep walking to catch deeper nests
                        walk(tcx, *nested_ty)
                    }
                    _ => walk(tcx, *inner_ty),
                }
            }
            ty::Tuple(tys) => tys.iter().any(|t| walk(tcx, t)),
            ty::Adt(_, args) => args.iter().any(|arg| match arg.kind() {
                ty::GenericArgKind::Type(t) => walk(tcx, t),
                _ => false,
            }),
            _ => false,
        }
    }

    walk(tcx, ty)
}
