//! This pass transforms derefs of Box into a deref of the pointer inside Box.
//!
//! Box is not actually a pointer so it is incorrect to dereference it directly.

use crate::MirPass;
use rustc_hir::def_id::DefId;
use rustc_index::vec::Idx;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};

/// Constructs the types used when accessing a Box's pointer
pub fn build_ptr_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointee: Ty<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
) -> (Ty<'tcx>, Ty<'tcx>, Ty<'tcx>) {
    let substs = tcx.mk_substs(&[pointee.into()]);
    let unique_ty = tcx.type_of(unique_did).subst(tcx, substs);
    let nonnull_ty = tcx.type_of(nonnull_did).subst(tcx, substs);
    let ptr_ty = tcx.mk_imm_ptr(pointee);

    (unique_ty, nonnull_ty, ptr_ty)
}

/// Constructs the projection needed to access a Box's pointer
pub fn build_projection<'tcx>(
    unique_ty: Ty<'tcx>,
    nonnull_ty: Ty<'tcx>,
    ptr_ty: Ty<'tcx>,
) -> [PlaceElem<'tcx>; 3] {
    [
        PlaceElem::Field(Field::new(0), unique_ty),
        PlaceElem::Field(Field::new(0), nonnull_ty),
        PlaceElem::Field(Field::new(0), ptr_ty),
    ]
}

struct ElaborateBoxDerefVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
    local_decls: &'a mut LocalDecls<'tcx>,
    patch: MirPatch<'tcx>,
}

impl<'tcx, 'a> MutVisitor<'tcx> for ElaborateBoxDerefVisitor<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(
        &mut self,
        place: &mut Place<'tcx>,
        context: visit::PlaceContext,
        location: Location,
    ) {
        let tcx = self.tcx;

        let base_ty = self.local_decls[place.local].ty;

        // Derefer ensures that derefs are always the first projection
        if place.projection.first() == Some(&PlaceElem::Deref) && base_ty.is_box() {
            let source_info = self.local_decls[place.local].source_info;

            let (unique_ty, nonnull_ty, ptr_ty) =
                build_ptr_tys(tcx, base_ty.boxed_ty(), self.unique_did, self.nonnull_did);

            let ptr_local = self.patch.new_internal(ptr_ty, source_info.span);

            self.patch.add_assign(
                location,
                Place::from(ptr_local),
                Rvalue::Use(Operand::Copy(
                    Place::from(place.local)
                        .project_deeper(&build_projection(unique_ty, nonnull_ty, ptr_ty), tcx),
                )),
            );

            place.local = ptr_local;
        }

        self.super_place(place, context, location);
    }
}

pub struct ElaborateBoxDerefs;

impl<'tcx> MirPass<'tcx> for ElaborateBoxDerefs {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if let Some(def_id) = tcx.lang_items().owned_box() {
            let unique_did = tcx.adt_def(def_id).non_enum_variant().fields[0].did;

            let Some(nonnull_def) = tcx.type_of(unique_did).subst_identity().ty_adt_def() else {
                span_bug!(tcx.def_span(unique_did), "expected Box to contain Unique")
            };

            let nonnull_did = nonnull_def.non_enum_variant().fields[0].did;

            let patch = MirPatch::new(body);

            let local_decls = &mut body.local_decls;

            let mut visitor =
                ElaborateBoxDerefVisitor { tcx, unique_did, nonnull_did, local_decls, patch };

            for (block, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
                visitor.visit_basic_block_data(block, data);
            }

            visitor.patch.apply(body);

            for debug_info in body.var_debug_info.iter_mut() {
                if let VarDebugInfoContents::Place(place) = &mut debug_info.value {
                    let mut new_projections: Option<Vec<_>> = None;
                    let mut last_deref = 0;

                    for (i, (base, elem)) in place.iter_projections().enumerate() {
                        let base_ty = base.ty(&body.local_decls, tcx).ty;

                        if elem == PlaceElem::Deref && base_ty.is_box() {
                            let new_projections = new_projections.get_or_insert_default();

                            let (unique_ty, nonnull_ty, ptr_ty) =
                                build_ptr_tys(tcx, base_ty.boxed_ty(), unique_did, nonnull_did);

                            new_projections.extend_from_slice(&base.projection[last_deref..]);
                            new_projections.extend_from_slice(&build_projection(
                                unique_ty, nonnull_ty, ptr_ty,
                            ));
                            new_projections.push(PlaceElem::Deref);

                            last_deref = i;
                        }
                    }

                    if let Some(mut new_projections) = new_projections {
                        new_projections.extend_from_slice(&place.projection[last_deref..]);
                        place.projection = tcx.mk_place_elems(&new_projections);
                    }
                }
            }
        } else {
            // box is not present, this pass doesn't need to do anything
        }
    }
}
