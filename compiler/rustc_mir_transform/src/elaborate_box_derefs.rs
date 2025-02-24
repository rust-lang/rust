//! This pass transforms derefs of Box into a deref of the pointer inside Box.
//!
//! Box is not actually a pointer so it is incorrect to dereference it directly.

use rustc_abi::FieldIdx;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::{Ty, TyCtxt};

use crate::patch::MirPatch;

/// Constructs the types used when accessing a Box's pointer
fn build_ptr_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointee: Ty<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
) -> (Ty<'tcx>, Ty<'tcx>, Ty<'tcx>) {
    let args = tcx.mk_args(&[pointee.into()]);
    let unique_ty = tcx.type_of(unique_did).instantiate(tcx, args);
    let nonnull_ty = tcx.type_of(nonnull_did).instantiate(tcx, args);
    let ptr_ty = Ty::new_imm_ptr(tcx, pointee);

    (unique_ty, nonnull_ty, ptr_ty)
}

/// Constructs the projection needed to access a Box's pointer
pub(super) fn build_projection<'tcx>(
    unique_ty: Ty<'tcx>,
    nonnull_ty: Ty<'tcx>,
) -> [PlaceElem<'tcx>; 2] {
    [PlaceElem::Field(FieldIdx::ZERO, unique_ty), PlaceElem::Field(FieldIdx::ZERO, nonnull_ty)]
}

struct ElaborateBoxDerefVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    unique_did: DefId,
    nonnull_did: DefId,
    local_decls: &'a mut LocalDecls<'tcx>,
    patch: MirPatch<'tcx>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for ElaborateBoxDerefVisitor<'a, 'tcx> {
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
        if let Some(PlaceElem::Deref) = place.projection.first()
            && let Some(boxed_ty) = base_ty.boxed_ty()
        {
            let source_info = self.local_decls[place.local].source_info;

            let (unique_ty, nonnull_ty, ptr_ty) =
                build_ptr_tys(tcx, boxed_ty, self.unique_did, self.nonnull_did);

            let ptr_local = self.patch.new_temp(ptr_ty, source_info.span);

            self.patch.add_assign(
                location,
                Place::from(ptr_local),
                Rvalue::Cast(
                    CastKind::Transmute,
                    Operand::Copy(
                        Place::from(place.local)
                            .project_deeper(&build_projection(unique_ty, nonnull_ty), tcx),
                    ),
                    ptr_ty,
                ),
            );

            place.local = ptr_local;
        }

        self.super_place(place, context, location);
    }
}

pub(super) struct ElaborateBoxDerefs;

impl<'tcx> crate::MirPass<'tcx> for ElaborateBoxDerefs {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // If box is not present, this pass doesn't need to do anything.
        let Some(def_id) = tcx.lang_items().owned_box() else { return };

        let unique_did = tcx.adt_def(def_id).non_enum_variant().fields[FieldIdx::ZERO].did;

        let Some(nonnull_def) = tcx.type_of(unique_did).instantiate_identity().ty_adt_def() else {
            span_bug!(tcx.def_span(unique_did), "expected Box to contain Unique")
        };

        let nonnull_did = nonnull_def.non_enum_variant().fields[FieldIdx::ZERO].did;

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

                for (base, elem) in place.iter_projections() {
                    let base_ty = base.ty(&body.local_decls, tcx).ty;

                    if let PlaceElem::Deref = elem
                        && let Some(boxed_ty) = base_ty.boxed_ty()
                    {
                        // Clone the projections before us, since now we need to mutate them.
                        let new_projections =
                            new_projections.get_or_insert_with(|| base.projection.to_vec());

                        let (unique_ty, nonnull_ty, ptr_ty) =
                            build_ptr_tys(tcx, boxed_ty, unique_did, nonnull_did);

                        new_projections.extend_from_slice(&build_projection(unique_ty, nonnull_ty));
                        // While we can't project into `NonNull<_>` in a basic block
                        // due to MCP#807, this is debug info where it's fine.
                        new_projections.push(PlaceElem::Field(FieldIdx::ZERO, ptr_ty));
                        new_projections.push(PlaceElem::Deref);
                    } else if let Some(new_projections) = new_projections.as_mut() {
                        // Keep building up our projections list once we've started it.
                        new_projections.push(elem);
                    }
                }

                // Store the mutated projections if we actually changed something.
                if let Some(new_projections) = new_projections {
                    place.projection = tcx.mk_place_elems(&new_projections);
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
