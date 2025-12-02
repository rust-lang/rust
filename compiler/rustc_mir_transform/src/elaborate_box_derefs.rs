//! This pass transforms derefs of Box into a deref of the pointer inside Box.
//!
//! Box is not actually a pointer so it is incorrect to dereference it directly.
//!
//! `ShallowInitBox` being a device for drop elaboration to understand deferred assignment to box
//! contents, we do not need this any more on runtime MIR.

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_index::{IndexVec, indexvec};
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Ty, TyCtxt};

use crate::patch::MirPatch;

/// Constructs the types used when accessing a Box's pointer
fn build_ptr_tys<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointee: Ty<'tcx>,
    unique_def: ty::AdtDef<'tcx>,
    nonnull_def: ty::AdtDef<'tcx>,
) -> (Ty<'tcx>, Ty<'tcx>, Ty<'tcx>) {
    let args = tcx.mk_args(&[pointee.into()]);
    let unique_ty = Ty::new_adt(tcx, unique_def, args);
    let nonnull_ty = Ty::new_adt(tcx, nonnull_def, args);
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
    unique_def: ty::AdtDef<'tcx>,
    nonnull_def: ty::AdtDef<'tcx>,
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
                build_ptr_tys(tcx, boxed_ty, self.unique_def, self.nonnull_def);

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

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);

        let tcx = self.tcx;
        let source_info = stmt.source_info;

        if let StatementKind::Assign(box (_, ref mut rvalue)) = stmt.kind
            && let Rvalue::ShallowInitBox(ref mut mutptr_to_u8, pointee) = *rvalue
            && let ty::Adt(box_adt, box_args) = Ty::new_box(tcx, pointee).kind()
        {
            let args = tcx.mk_args(&[pointee.into()]);
            let (unique_ty, nonnull_ty, ptr_ty) =
                build_ptr_tys(tcx, pointee, self.unique_def, self.nonnull_def);
            let adt_kind = |def: ty::AdtDef<'tcx>, args| {
                Box::new(AggregateKind::Adt(def.did(), VariantIdx::ZERO, args, None, None))
            };
            let zst = |ty| {
                Operand::Constant(Box::new(ConstOperand {
                    span: source_info.span,
                    user_ty: None,
                    const_: Const::zero_sized(ty),
                }))
            };

            let constptr = self.patch.new_temp(ptr_ty, source_info.span);
            self.patch.add_assign(
                location,
                constptr.into(),
                Rvalue::Cast(CastKind::Transmute, mutptr_to_u8.clone(), ptr_ty),
            );

            let nonnull = self.patch.new_temp(nonnull_ty, source_info.span);
            self.patch.add_assign(
                location,
                nonnull.into(),
                Rvalue::Aggregate(
                    adt_kind(self.nonnull_def, args),
                    indexvec![Operand::Move(constptr.into())],
                ),
            );

            let unique = self.patch.new_temp(unique_ty, source_info.span);
            let phantomdata_ty =
                self.unique_def.non_enum_variant().fields[FieldIdx::ONE].ty(tcx, args);
            self.patch.add_assign(
                location,
                unique.into(),
                Rvalue::Aggregate(
                    adt_kind(self.unique_def, args),
                    indexvec![Operand::Move(nonnull.into()), zst(phantomdata_ty)],
                ),
            );

            let global_alloc_ty =
                box_adt.non_enum_variant().fields[FieldIdx::ONE].ty(tcx, box_args);
            *rvalue = Rvalue::Aggregate(
                adt_kind(*box_adt, box_args),
                indexvec![Operand::Move(unique.into()), zst(global_alloc_ty)],
            );
        }
    }
}

pub(super) struct ElaborateBoxDerefs;

impl<'tcx> crate::MirPass<'tcx> for ElaborateBoxDerefs {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // If box is not present, this pass doesn't need to do anything.
        let Some(def_id) = tcx.lang_items().owned_box() else { return };

        let unique_did = tcx.adt_def(def_id).non_enum_variant().fields[FieldIdx::ZERO].did;

        let Some(unique_def) = tcx.type_of(unique_did).instantiate_identity().ty_adt_def() else {
            span_bug!(tcx.def_span(unique_did), "expected Box to contain Unique")
        };

        let nonnull_did = unique_def.non_enum_variant().fields[FieldIdx::ZERO].did;

        let Some(nonnull_def) = tcx.type_of(nonnull_did).instantiate_identity().ty_adt_def() else {
            span_bug!(tcx.def_span(nonnull_did), "expected Unique to contain Nonnull")
        };

        let patch = MirPatch::new(body);

        let local_decls = &mut body.local_decls;

        let mut visitor =
            ElaborateBoxDerefVisitor { tcx, unique_def, nonnull_def, local_decls, patch };

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
                            build_ptr_tys(tcx, boxed_ty, unique_def, nonnull_def);

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
