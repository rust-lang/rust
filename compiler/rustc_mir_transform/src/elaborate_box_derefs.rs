//! This pass transforms derefs of Box into a deref of the pointer inside Box.
//!
//! Box is not actually a pointer so it is incorrect to dereference it directly.

use crate::MirPass;
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_index::vec::Idx;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

/// Constructs the types used when accessing a Box's pointer
pub fn build_ptr_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointee: Ty<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
    ranged_did: DefId,
    ranged_range: ty::GenericArg<'tcx>,
) -> [Ty<'tcx>; 4] {
    let substs = tcx.intern_substs(&[pointee.into()]);
    let unique_ty = tcx.bound_type_of(unique_did).subst(tcx, substs);
    let nonnull_ty = tcx.bound_type_of(nonnull_did).subst(tcx, substs);
    let ptr_ty = tcx.mk_imm_ptr(pointee);

    let substs = tcx.intern_substs(&[ptr_ty.into(), ranged_range]);
    let ranged_ty = tcx.bound_type_of(ranged_did).subst(tcx, substs);

    [unique_ty, nonnull_ty, ranged_ty, ptr_ty]
}

// Constructs the projection needed to access a Box's pointer
pub fn build_projection<'tcx>(
    [unique_ty, nonnull_ty, ranged_ty, ptr_ty]: [Ty<'tcx>; 4],
) -> [PlaceElem<'tcx>; 4] {
    [
        PlaceElem::Field(Field::new(0), unique_ty),
        PlaceElem::Field(Field::new(0), nonnull_ty),
        PlaceElem::Field(Field::new(0), ranged_ty),
        PlaceElem::Field(Field::new(0), ptr_ty),
    ]
}

struct ElaborateBoxDerefVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
    ranged_did: DefId,
    ranged_range: ty::GenericArg<'tcx>,
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

            let field_types = build_ptr_tys(
                tcx,
                base_ty.boxed_ty(),
                self.unique_did,
                self.nonnull_did,
                self.ranged_did,
                self.ranged_range,
            );

            let ptr_local = self.patch.new_internal(field_types[3], source_info.span);

            self.patch.add_assign(
                location,
                Place::from(ptr_local),
                Rvalue::Use(Operand::Copy(
                    Place::from(place.local).project_deeper(&build_projection(field_types), tcx),
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

            let Some(unique_def) = tcx.type_of(unique_did).ty_adt_def() else {
                span_bug!(tcx.def_span(unique_did), "expected Box to contain Unique")
            };

            let nonnull_did = unique_def.non_enum_variant().fields[0].did;
            let ranged_did = tcx.require_lang_item(LangItem::Ranged, Some(body.span));
            let Some(nonnull_def) = tcx.type_of(nonnull_did).ty_adt_def() else {
                span_bug!(tcx.def_span(nonnull_did), "expected Unique to contain NonNull")
            };
            let ranged_field = nonnull_def.non_enum_variant().fields[0].did;
            let ty::Adt(_, substs) = tcx.type_of(ranged_field).kind() else {
                span_bug!(tcx.def_span(ranged_field), "expected NonNull to contain Ranged")
            };
            let ranged_range = substs[1];

            let patch = MirPatch::new(body);

            let local_decls = &mut body.local_decls;

            let mut visitor = ElaborateBoxDerefVisitor {
                tcx,
                unique_did,
                nonnull_did,
                ranged_did,
                ranged_range,
                local_decls,
                patch,
            };

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

                            let field_types = build_ptr_tys(
                                tcx,
                                base_ty.boxed_ty(),
                                unique_did,
                                nonnull_did,
                                ranged_did,
                                ranged_range,
                            );

                            new_projections.extend_from_slice(&base.projection[last_deref..]);
                            new_projections.extend_from_slice(&build_projection(field_types));
                            new_projections.push(PlaceElem::Deref);

                            last_deref = i;
                        }
                    }

                    if let Some(mut new_projections) = new_projections {
                        new_projections.extend_from_slice(&place.projection[last_deref..]);
                        place.projection = tcx.intern_place_elems(&new_projections);
                    }
                }
            }
        } else {
            // box is not present, this pass doesn't need to do anything
        }
    }
}
