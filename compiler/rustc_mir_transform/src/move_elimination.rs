//! Eliminates copies and moves by unifying MIR places whose allocation ranges
//! are disjoint.
//!
//! See RFC 3943 for the local-lifetime semantics that make this optimization
//! possible.
//!
//! # Motivation
//!
//! MIR building can insert a lot of redundant copies, and Rust code in general
//! often tends to move values around a lot. The result is a lot of assignments
//! of the form `dest = {move} src;` in MIR. MIR building for constants in
//! particular tends to create additional locals that are only used inside a
//! single block to shuffle a value around unnecessarily.
//!
//! Additionally, Rust constructs nested aggregates by repeatedly moving values
//! into fields. For example, a function may build an inner value in a local,
//! move it into an outer aggregate, then move that aggregate into the caller's
//! destination. If these intermediate source and destination places have
//! different addresses, each layer needs an actual copy or move of the bytes.
//!
//! LLVM cannot remove these copies when both the source and destination
//! addresses are observed because merging the allocations would be an
//! observable change: the program could see that two addresses which were
//! previously distinct have become the same. This pass removes the copies
//! earlier, while MIR still has the information needed to prove that the two
//! allocation ranges do not overlap.
//!
//! # Optimization
//!
//! The basis of this optimization is place unification. If the source and
//! destination of an assignment have the same address, then the assignment is a
//! no-op. The same idea applies to aggregate construction: if a field operand
//! is already located at the corresponding field of the destination, then the
//! aggregate assignment does not need to copy those fields.
//!
//! The pass represents each unification as a mapping from a local to the place
//! that should replace it. Mappings are transitive, so `_3` can be resolved
//! through `_2.1` to `_1.0.1` if earlier mappings established those
//! relationships.
//!
//! The mapping is built by scanning the MIR for assignment statements. For
//! simple `Use` assignments, it tries to unify the source and destination
//! places. For `Aggregate` assignments, it tries to map each field operand to
//! the corresponding field in the assignment destination. Once all mappings
//! have been chosen, they are applied with one rewrite pass over the body.
//!
//! # Constraints
//!
//! Adding a mapping must preserve these conditions:
//!
//! * At least one side of the candidate pair must be a bare local. The pass can
//!   map a local to a place with projections, but it cannot map between two
//!   places that both already have projections.
//!
//! * Any projections in the mapped place must be stable everywhere the local is
//!   used. `Deref` and `Index` projections are rejected because they may refer
//!   to different memory at different points in the function.
//!
//! * The allocation ranges of the source and destination places must not
//!   overlap. This is checked using `PreciseLiveness`, which computes the
//!   points where each local must have a distinct allocation. The non-overlap
//!   proof is required so that the operational semantics can allow both places
//!   to have the same address.
//!
//! * Special-use locals such as arguments and the return place must keep their
//!   roles. Temps may be mapped into an argument or return place, but two
//!   special-use locals are not mapped into each other.
//!
//! * Some locals are used in contexts where projections cannot be added, such
//!   as `Index` projections. These locals may only be replaced by another bare
//!   local.
//!
//! # Storage reconstruction
//!
//! The original `StorageLive` and `StorageDead` statements no longer describe
//! the merged liveness produced by unification, so they are removed and rebuilt
//! from the liveness matrix when lifetime markers are emitted. This is done for
//! all locals, even ones that have not been merged, which has the additional
//! benefit of tightening the storage lifetime passed to LLVM.
//!
//! # Aliasing fixup
//!
//! MIR assignments currently require source and destination places not to
//! overlap for types that are not treated as scalars in codegen. After local
//! unification, some assignments may violate that invariant, so a final phase
//! rewrites them into a form codegen can handle. For each assignment:
//!
//! * Self-assignments, such as `_1 = _1`, are deleted.
//!
//! * Simple `Use` assignments whose source and destination overlap but are not
//!   identical are routed through a temporary: the source is read into the
//!   temporary first, and the temporary is then moved into the destination.
//!
//! * Aggregate assignments with any operand that aliases the destination are
//!   decomposed into per-field assignments. Field self-assignments are dropped.
//!   Other aliasing fields are read into temporaries first, then all
//!   destination fields are written. For enum aggregates the discriminant is
//!   set after fields are written.
//!
//! * Other rvalues, such as `Repeat` and `Cast`, are hoisted into a temporary
//!   if any place they access aliases the destination.
//!
//! * Rvalues that operate only on scalar types, such as binary and unary ops,
//!   `Discriminant`, `Ref`, and `RawPtr`, are left untouched because their
//!   codegen does not rely on the no-aliasing assumption.

use rustc_abi::{ExternAbi, FieldIdx, VariantIdx};
use rustc_const_eval::util::most_packed_projection;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::interval::SparseIntervalMatrix;
use rustc_middle::mir::visit::{MutVisitor, NonUseContext, PlaceContext, VisitPlacesWith, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_mir_dataflow::impls::{
    SplitPointEffect, SplitPointIndex, dump_liveness_matrix, liveness_matrix,
};
use rustc_mir_dataflow::points::DenseLocationMap;
use tracing::{debug, trace};

use crate::patch::MirPatch;

pub(super) struct MoveElimination;

impl<'tcx> crate::MirPass<'tcx> for MoveElimination {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2 && sess.opts.unstable_opts.mir_move_elimination
    }

    #[tracing::instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        trace!(?def_id);

        let points = DenseLocationMap::new(body);
        let mut liveness_matrix =
            liveness_matrix(tcx, body, &points, Some("MoveElimination.liveness"));

        dump_liveness_matrix(tcx, body, "MoveElimination.pre-liveness", &points, &liveness_matrix);

        let unprojectable_locals = UnprojectableLocals::find(body);
        trace!(?unprojectable_locals);

        let rust_call_tuples = find_rust_call_tuples(tcx, body);
        trace!(?rust_call_tuples);

        let remapped_locals = PlaceUnification::run(
            tcx,
            body,
            &mut liveness_matrix,
            unprojectable_locals,
            rust_call_tuples,
        );

        apply_mappings(tcx, body, &remapped_locals);

        dump_liveness_matrix(tcx, body, "MoveElimination.post-liveness", &points, &liveness_matrix);

        if tcx.sess.emit_lifetime_markers() {
            reconstruct_storage(body, &points, &liveness_matrix);
        }

        apply_alias_fixup(tcx, body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unprojectable locals

/// Set of locals which can only be replaced with another local, instead of
/// an arbitrary place. This is usually because it is used directly as a
/// `Local` outside of a place (e.g. `Index` projections).
#[derive(Debug)]
struct UnprojectableLocals {
    locals: DenseBitSet<Local>,
}

impl UnprojectableLocals {
    fn find(body: &Body<'_>) -> DenseBitSet<Local> {
        let mut out = Self { locals: DenseBitSet::new_empty(body.local_decls.len()) };

        // Arguments and return places have fixed roles and cannot be replaced
        // with projected locals.
        out.locals.insert(RETURN_PLACE);
        for arg in body.args_iter() {
            out.locals.insert(arg);
        }

        out.visit_body(body);
        out.locals
    }
}

impl<'tcx> Visitor<'tcx> for UnprojectableLocals {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // We can't add more projections before a first position Deref projection.
        if place.is_indirect() {
            trace!(
                "unprojectable local {:?} due to use as deref base at {location:?}",
                place.local
            );
            self.locals.insert(place.local);
        }

        // Only call visit_local for projections, not the base local.
        self.visit_projection(place.as_ref(), context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        // Ignore uses in storage statements, we're going to remove all of those
        // anyways.
        if let PlaceContext::NonUse(NonUseContext::StorageLive | NonUseContext::StorageDead) =
            context
        {
            return;
        }

        // If this is reached, it means that this is a bare local used outside
        // of a place, which means it cannot be replaced with a projection of
        // another local.
        trace!("unprojectable local {local:?} at {location:?} ({context:?})");
        self.locals.insert(local);
    }
}

////////////////////////////////////////////////////////////////////////////////
// "rust-call" tuple handling

/// Search for tuple locals passed to calls using the "rust-call" ABI.
///
/// For rust-call ABI calls, caller-side MIR passes the logical arguments as a
/// tuple operand. We want to avoid remapping other locals into fields of that
/// tuple, especially if one of those locals is borrowed.
///
/// Since the tuple itself is never borrowed, it is trivial for LLVM alias
/// analysis to see that accesses to one argument do not affect the others, but
/// merging the arguments into tuple fields from the start can hide that
/// independence.
fn find_rust_call_tuples<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> DenseBitSet<Local> {
    let mut rust_call_tuples = DenseBitSet::new_empty(body.local_decls.len());

    for block in body.basic_blocks.iter() {
        let terminator = block.terminator();
        let (func, args) = match &terminator.kind {
            TerminatorKind::Call { func, args, .. }
            | TerminatorKind::TailCall { func, args, .. } => (func, args),
            _ => continue,
        };

        let sig = func.ty(&body.local_decls, tcx).fn_sig(tcx);
        if sig.abi() != ExternAbi::RustCall {
            continue;
        }

        let arg_tuple = args.last().expect("rust-call ABI requires a tuple argument");
        let (Operand::Copy(place) | Operand::Move(place)) = arg_tuple.node else {
            continue;
        };
        if let Some(local) = place.as_local() {
            rust_call_tuples.insert(local);
        }
    }

    rust_call_tuples
}

////////////////////////////////////////////////////////////////////////////////
// Local unification

struct PlaceUnification<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    liveness_matrix: &'a mut SparseIntervalMatrix<Local, SplitPointIndex>,
    unprojectable_locals: DenseBitSet<Local>,
    rust_call_tuples: DenseBitSet<Local>,
    remapped_locals: IndexVec<Local, Option<Place<'tcx>>>,
}

impl<'tcx> PlaceUnification<'_, 'tcx> {
    fn run(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        liveness_matrix: &mut SparseIntervalMatrix<Local, SplitPointIndex>,
        unprojectable_locals: DenseBitSet<Local>,
        rust_call_tuples: DenseBitSet<Local>,
    ) -> IndexVec<Local, Option<Place<'tcx>>> {
        let mut visitor = PlaceUnification {
            tcx,
            body,
            liveness_matrix,
            unprojectable_locals,
            rust_call_tuples,
            remapped_locals: IndexVec::from_elem_n(None, body.local_decls.len()),
        };
        visitor.visit_body(body);

        // Finalize the mappings by transitively resolving all locals to their
        // new final place.
        for local in visitor.remapped_locals.indices() {
            if let Some(place) = visitor.remapped_locals[local] {
                let place = visitor.resolve_place(place);
                visitor.remapped_locals[local] = Some(place);
                trace!("Remapped {local:?} to {place:?}");
            }
        }

        visitor.remapped_locals
    }

    #[tracing::instrument(ret, level = "trace", skip(self))]
    fn resolve_place(&self, mut place: Place<'tcx>) -> Place<'tcx> {
        while let Some(new_place) = self.remapped_locals[place.local] {
            place = new_place.project_deeper(place.projection, self.tcx);
        }
        place
    }

    #[tracing::instrument(ret, level = "trace", skip(self))]
    fn can_unify_places(&self, a: Place<'tcx>, b: Place<'tcx>) -> Option<(Local, Place<'tcx>)> {
        let a = self.resolve_place(a);
        let b = self.resolve_place(b);

        if a.local == b.local {
            if a.projection != b.projection {
                trace!("cannot unify same local with different projections");
            }
            return None;
        }

        if self.rust_call_tuples.contains(a.local) || self.rust_call_tuples.contains(b.local) {
            trace!("cannot unify {a:?} and {b:?} involving a rust-call tuple argument");
            return None;
        }

        let (local, place) = match (a.as_local(), b.as_local()) {
            (None, None) => {
                trace!("cannot unify 2 places that both have projections");
                return None;
            }
            (None, Some(b)) => {
                if self.unprojectable_locals.contains(b) {
                    trace!("cannot unify {b:?} which cannot be projected");
                    return None;
                }
                (b, a)
            }
            (Some(a), None) => {
                if self.unprojectable_locals.contains(a) {
                    trace!("cannot unify {a:?} which cannot be projected");
                    return None;
                }
                (a, b)
            }
            (Some(a), Some(b)) => match (self.body.local_kind(a), self.body.local_kind(b)) {
                (
                    LocalKind::Arg | LocalKind::ReturnPointer,
                    LocalKind::Arg | LocalKind::ReturnPointer,
                ) => {
                    trace!("cannot unify {a:?} and {b:?} which are both arguments or return place");
                    return None;
                }
                (LocalKind::Arg | LocalKind::ReturnPointer, LocalKind::Temp) => (b, a.into()),
                (LocalKind::Temp, _) => (a, b.into()),
            },
        };

        if most_packed_projection(self.tcx, &self.body.local_decls, place).is_some() {
            trace!("cannot unify {place:?} which has packed field projections");
            return None;
        }

        if !self.liveness_matrix.disjoint_rows(local, place.local) {
            trace!("cannot unify {a:?} and {b:?} which have overlapping live ranges");
            return None;
        }

        // FIXME(#112651): This can be removed afterwards.
        let local_ty = self.body.local_decls[local].ty;
        let place_ty = place.ty(&self.body.local_decls, self.tcx).ty;
        if local_ty != place_ty {
            trace!(
                "cannot unify {a:?} and {b:?} which have different types due to subtyping ({local_ty:?} vs {place_ty:?})"
            );
            return None;
        }

        Some((local, place))
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn remap_local(&mut self, local: Local, place: Place<'tcx>) {
        self.remapped_locals[local] = Some(place);

        self.liveness_matrix.union_rows(local, place.local);
        self.liveness_matrix.clear_row(local);

        // If the original local was unprojectable then this now also applies to
        // the mapped local.
        if self.unprojectable_locals.contains(local) {
            debug_assert!(place.projection.is_empty());
            self.unprojectable_locals.insert(place.local);
        }
    }

    fn visit_aggregate_assign(
        &mut self,
        dest: Place<'tcx>,
        project_field: impl Fn(TyCtxt<'tcx>, Place<'tcx>, FieldIdx, Ty<'tcx>) -> Place<'tcx>,
        operands: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location,
    ) {
        // Attempt to unify each field operand with the corresponding field in
        // the destination place.
        let mut candidates = vec![];
        for (idx, operand) in operands.iter_enumerated() {
            let (Operand::Copy(src) | Operand::Move(src)) = *operand else {
                continue;
            };
            let Some(src) = src.as_local() else {
                continue;
            };
            let dest = project_field(self.tcx, dest, idx, self.body.local_decls[src].ty);
            trace!("Attempting to unify {dest:?} and {src:?} at {location:?}");
            if let Some((local, place)) = self.can_unify_places(dest, src.into()) {
                candidates.push((local, place));
            }
        }

        // Do the actual remapping *after* checking for live range overlaps.
        // This is necessary because the input operands necessarily have
        // overlapping live ranges.
        for (local, place) in candidates {
            self.remap_local(local, place);
        }
    }
}

/// Since we are replacing all uses of a local with another place, we need to
/// ensure that the projections on that place are stable no matter where it is
/// used in the body. Additional this local may be used in debuginfo, so ensure
/// that the projections are compatible with usage in debuginfo.
fn check_projections(place: Place<'_>) -> bool {
    place.projection.iter().all(|elem| elem.is_stable_offset() && elem.can_use_in_debuginfo())
}

impl<'tcx> Visitor<'tcx> for PlaceUnification<'_, 'tcx> {
    fn visit_assign(&mut self, dest: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        if !check_projections(*dest) {
            return;
        }
        match rvalue {
            Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                if !check_projections(*src) {
                    return;
                }

                trace!("Attempting to unify {dest:?} and {src:?} at {location:?}");
                if let Some((local, place)) = self.can_unify_places(*src, *dest) {
                    self.remap_local(local, place);
                }
            }
            Rvalue::Aggregate(aggregate_kind, operands) => match *aggregate_kind {
                AggregateKind::Array(_) => self.visit_aggregate_assign(
                    *dest,
                    |tcx, place, field_idx, _field_ty| {
                        place.project_deeper(
                            &[PlaceElem::ConstantIndex {
                                offset: field_idx.as_u32().into(),
                                min_length: field_idx.as_u32() as u64 + 1,
                                from_end: false,
                            }],
                            tcx,
                        )
                    },
                    operands,
                    location,
                ),
                AggregateKind::Tuple => self.visit_aggregate_assign(
                    *dest,
                    |tcx, place, field_idx, field_ty| {
                        place.project_deeper(&[PlaceElem::Field(field_idx, field_ty)], tcx)
                    },
                    operands,
                    location,
                ),
                AggregateKind::Adt(_, _, _, _, Some(union_field_idx)) => {
                    debug_assert_eq!(operands.len(), 1);
                    self.visit_aggregate_assign(
                        *dest,
                        |tcx, place, _, field_ty| {
                            place
                                .project_deeper(&[PlaceElem::Field(union_field_idx, field_ty)], tcx)
                        },
                        operands,
                        location,
                    )
                }
                AggregateKind::Adt(adt_did, var_idx, _, _, None) => {
                    let def = self.tcx.adt_def(adt_did);
                    if def.repr().simd() {
                        // MCP#838 banned projections into SIMD types.
                        return;
                    }
                    self.visit_aggregate_assign(
                        *dest,
                        |tcx, place, field_idx, field_ty| {
                            if def.is_enum() {
                                place.project_deeper(
                                    &[
                                        PlaceElem::Downcast(None, var_idx),
                                        PlaceElem::Field(field_idx, field_ty),
                                    ],
                                    tcx,
                                )
                            } else {
                                place.project_deeper(&[PlaceElem::Field(field_idx, field_ty)], tcx)
                            }
                        },
                        operands,
                        location,
                    )
                }
                _ => {}
            },
            _ => {}
        };
    }
}

////////////////////////////////////////////////////////////////////////////////
// Apply place mappings to the MIR body.

fn apply_mappings<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    remapped_locals: &IndexVec<Local, Option<Place<'tcx>>>,
) {
    let mut rewriter = PlaceUpdater { tcx, remapped_locals };
    rewriter.visit_body_preserves_cfg(body);
}

struct PlaceUpdater<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    remapped_locals: &'a IndexVec<Local, Option<Place<'tcx>>>,
}

impl<'tcx> MutVisitor<'tcx> for PlaceUpdater<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        if let Some(new_place) = self.remapped_locals[*local] {
            trace!("replacing {local:?} with {new_place:?} at {location:?} ({context:?})");
            *local = new_place.as_local().expect("mapped place shouldn't have projections");
        }
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some(new_place) = self.remapped_locals[place.local] {
            trace!("replacing {place:?} with {new_place:?} at {location:?} ({context:?})");
            *place = new_place.project_deeper(place.projection, self.tcx)
        }

        // Only call visit_local for projections, not the base local.
        if let Some(new_projection) = self.process_projection(&place.projection, location) {
            place.projection = self.tcx().mk_place_elems(&new_projection);
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match statement.kind {
            // Remove *all* storage statements. These are rebuilt from liveness
            // information later. Also, since we've preserved StorageDead in
            // unwind paths until now, we will want to remove those since they
            // hurt LLVM's codegen.
            StatementKind::StorageDead(_) | StatementKind::StorageLive(_) => {
                statement.make_nop(true);
                return;
            }
            _ => {}
        }

        self.super_statement(statement, location);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Storage reconstruction

/// Helper function to split a critical edge if necessary.
fn get_or_split_edge<'tcx>(
    patcher: &mut MirPatch<'tcx>,
    body: &Body<'tcx>,
    split_edges: &mut FxHashMap<(BasicBlock, BasicBlock), BasicBlock>,
    pred: BasicBlock,
    succ: BasicBlock,
) -> BasicBlock {
    if let Some(&split_bb) = split_edges.get(&(pred, succ)) {
        return split_bb;
    }
    let source_info = body.basic_blocks[pred].terminator().source_info;
    let split_bb = patcher.new_block(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::Goto { target: succ },
            attributes: ThinVec::new(),
        }),
        body.basic_blocks[succ].is_cleanup,
    ));
    patcher.mutate_terminator(body, pred, |kind| {
        kind.successors_mut(|t| {
            if *t == succ {
                *t = split_bb;
            }
        });
    });
    split_edges.insert((pred, succ), split_bb);
    split_bb
}

/// Don't insert storage statements in cleanup blocks and in unreachable blocks.
fn should_insert_storage<'tcx>(block_data: &BasicBlockData<'tcx>) -> bool {
    !block_data.is_cleanup && !matches!(block_data.terminator().kind, TerminatorKind::Unreachable)
}

/// Re-constructs storage statements for all locals.
fn reconstruct_storage<'tcx>(
    body: &mut Body<'tcx>,
    points: &DenseLocationMap,
    liveness_matrix: &SparseIntervalMatrix<Local, SplitPointIndex>,
) {
    let mut patcher = MirPatch::new(body);
    let mut split_edges: FxHashMap<(BasicBlock, BasicBlock), BasicBlock> = Default::default();

    for local in body.local_decls.indices() {
        // Arguments and return values don't use storage statements.
        match body.local_kind(local) {
            LocalKind::Arg | LocalKind::ReturnPointer => continue,
            LocalKind::Temp => {}
        }

        // Ignore dead locals.
        let Some(row) = liveness_matrix.row(local) else { continue };
        if row.is_empty() {
            continue;
        }

        // Helper functions to emit storage statements in block predecessors and
        // successors.
        let mut emit_storage_live_in_preds =
            |body: &mut Body<'tcx>,
             patcher: &mut MirPatch<'tcx>,
             local: Local,
             block: BasicBlock| {
                for &pred in &body.basic_blocks.predecessors()[block].clone() {
                    // If the local is live at any point in the predecessor's
                    // terminator then no StorageLive is needed.
                    let term_early =
                        SplitPointIndex::new(points.terminator(pred), SplitPointEffect::Early);
                    let term_late =
                        SplitPointIndex::new(points.terminator(pred), SplitPointEffect::Late);
                    if !row.intersects_range(term_early..=term_late) {
                        // The local must be live on at least one predecessor,
                        // so if this is the only one then there is nothing to
                        // do.
                        debug_assert!(body.basic_blocks.predecessors()[block].len() > 1);

                        // If the predecessor block has multiple successors then
                        // we need to split the critical edge before inserting
                        // StorageLive, otherwise the local would end up live on
                        // paths where it is supposed to be dead.
                        let loc = if body.basic_blocks[pred].terminator().successors().count() > 1 {
                            get_or_split_edge(patcher, body, &mut split_edges, pred, block)
                                .start_location()
                        } else {
                            body.terminator_loc(pred)
                        };
                        patcher.add_statement(loc, StatementKind::StorageLive(local));
                    }
                }
            };
        let emit_storage_dead_in_succs =
            |body: &mut Body<'tcx>,
             patcher: &mut MirPatch<'tcx>,
             local: Local,
             block: BasicBlock| {
                for succ in body.basic_blocks[block].terminator().successors() {
                    if !should_insert_storage(&body.basic_blocks[succ]) {
                        continue;
                    }

                    if !row.contains(SplitPointIndex::new(
                        points.entry_point(succ),
                        SplitPointEffect::Early,
                    )) {
                        // We don't care about critical edges here: if the local
                        // is already dead in the successor then it doesn't
                        // matter if we emit a redundant StorageDead.

                        patcher.add_statement(
                            succ.start_location(),
                            StatementKind::StorageDead(local),
                        );
                    }
                }
            };

        // Iterate through the live range of the local and insert `StorageLive`
        // and `StorageDead` at the points where it transitions from dead to
        // live and vice versa.
        //
        // Note that the range here is an *inclusive range*.
        for range in row.iter_intervals() {
            let start_block = points.to_location(range.start.point()).block;
            let end_block = points.to_location(range.last.point()).block;

            // If the live range starts at the `Early` point then it means that
            // the value came from a predecessor block. A write from the first
            // statement would happen at the `Late` point instead.
            if should_insert_storage(&body.basic_blocks[start_block]) {
                if range.start
                    == SplitPointIndex::new(
                        points.entry_point(start_block),
                        SplitPointEffect::Early,
                    )
                {
                    // If the local is dead at the end of any predecessor block then
                    // emit a `StorageLive` before the terminator.
                    emit_storage_live_in_preds(body, &mut patcher, local, start_block);
                } else {
                    // Otherwise just add `StorageLive` before the statement that
                    // starts the live range.
                    patcher.add_statement(
                        points.to_location(range.start.point()),
                        StatementKind::StorageLive(local),
                    );
                }
            }

            // The live range may span multiple blocks because
            // `SparseIntervalMatrix` will coalesce adjacent ranges. If this
            // happens then we need to repeat the start of block logic (see
            // above) and end of block logic (see below) at each block boundary.
            let mut current_block = start_block;
            debug_assert!(start_block <= end_block);
            while current_block != end_block {
                if should_insert_storage(&body.basic_blocks[current_block]) {
                    emit_storage_dead_in_succs(body, &mut patcher, local, current_block);
                }
                current_block = BasicBlock::from_usize(current_block.index() + 1);
                if should_insert_storage(&body.basic_blocks[current_block]) {
                    emit_storage_live_in_preds(body, &mut patcher, local, current_block);
                }
            }

            // We need to insert `StorageDead` after the last statement that
            // uses a local. If this is a terminator then we need to instead
            // insert it at the start of every successor block where the local
            // is dead on entry.
            if should_insert_storage(&body.basic_blocks[end_block]) {
                if range.last.point() == points.terminator(end_block) {
                    emit_storage_dead_in_succs(body, &mut patcher, local, current_block);
                } else {
                    // Don't emit StorageDead in cleanup blocks.
                    if !body.basic_blocks[end_block].is_cleanup {
                        patcher.add_statement(
                            points.to_location(range.last.point()).successor_within_block(),
                            StatementKind::StorageDead(local),
                        );
                    }
                }
            }
        }
    }

    patcher.apply(body);
}

////////////////////////////////////////////////////////////////////////////////
// Aliasing assignment fixup
//
// MIR assignments currently do not allow source and destination to alias, so
// fix this in post-processing.

fn apply_alias_fixup<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let mut patcher = MirPatch::new(body);
    let mut fixup = AliasFixup { tcx, local_decls: &body.local_decls, patcher: &mut patcher };
    for (block, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        fixup.visit_basic_block_data(block, data);
    }
    patcher.apply(body);
}

fn places_alias<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &IndexVec<Local, LocalDecl<'tcx>>,
    a: Place<'tcx>,
    b: Place<'tcx>,
) -> bool {
    // Indirect places don't overlap because we assume they didn't overlap in
    // the input MIR.
    if a.local != b.local || a.is_indirect_first_projection() || b.is_indirect_first_projection() {
        return false;
    }

    for ((prefix, elem_a), (_, elem_b)) in a.iter_projections().zip(b.iter_projections()) {
        // Continue until we find the first mismatching projection.
        if elem_a == elem_b {
            continue;
        }

        match (elem_a, elem_b) {
            // Disjoint fields don't alias except if they are union fields.
            (PlaceElem::Field(_, _), PlaceElem::Field(_, _)) => {
                let ty = prefix.ty(local_decls, tcx).ty;
                return ty.is_union();
            }

            // Disjoint slice elements don't alias.
            (
                PlaceElem::ConstantIndex { offset: offset_a, min_length: _, from_end: from_end_a },
                PlaceElem::ConstantIndex { offset: offset_b, min_length: _, from_end: from_end_b },
            ) if from_end_a == from_end_b && offset_a != offset_b => {
                return false;
            }

            // Conservatively assume the places may alias.
            _ => return true,
        }
    }

    // If the projections are identical *or* one is a prefix of the other then
    // the places alias.
    true
}

struct AliasFixup<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a IndexVec<Local, LocalDecl<'tcx>>,
    patcher: &'a mut MirPatch<'tcx>,
}

impl<'tcx> AliasFixup<'_, 'tcx> {
    fn isolate_rvalue_to_local(
        &mut self,
        rvalue: Rvalue<'tcx>,
        source_info: SourceInfo,
        location: Location,
    ) -> Place<'tcx> {
        let ty = rvalue.ty(self.local_decls, self.tcx);
        let temp = Place::from(self.patcher.new_temp(ty, source_info.span));
        trace!("isolating {rvalue:?} to {temp:?} due to conflict");
        self.patcher.add_statement(location, StatementKind::StorageLive(temp.local));
        self.patcher.add_assign(location, Place::from(temp), rvalue);
        self.patcher.add_statement(
            location.successor_within_block(),
            StatementKind::StorageDead(temp.local),
        );
        temp
    }

    fn visit_aggregate_assign(
        &mut self,
        dest: Place<'tcx>,
        enum_variant: Option<VariantIdx>,
        project_field: impl Fn(TyCtxt<'tcx>, Place<'tcx>, FieldIdx, Ty<'tcx>) -> Place<'tcx>,
        operands: &IndexVec<FieldIdx, Operand<'tcx>>,
        source_info: SourceInfo,
        location: Location,
    ) {
        // Fast path: if no operand alias the destination, we're done.
        let has_any_alias = operands.iter().any(|op| match op {
            Operand::Copy(src) | Operand::Move(src) => {
                places_alias(self.tcx, self.local_decls, dest, *src)
            }
            Operand::Constant(_) | Operand::RuntimeChecks(_) => false,
        });
        if !has_any_alias {
            return;
        }

        debug!("splitting aggregate assignment at {location:?}");

        // Split into per-field assignments.
        let mut assignments = vec![];
        for (idx, op) in operands.iter_enumerated() {
            let field_ty = op.ty(self.local_decls, self.tcx);
            let dest_field = project_field(self.tcx, dest, idx, field_ty);

            let emit_op = match op {
                Operand::Copy(src) | Operand::Move(src) => {
                    if *src == dest_field {
                        // Skip identity assignments.
                        continue;
                    } else if places_alias(self.tcx, self.local_decls, dest, *src) {
                        // Partial alias: hoist the source to a temp first so
                        // the per-field write no longer overlaps the dest.
                        Operand::Move(self.isolate_rvalue_to_local(
                            Rvalue::Use(op.clone(), WithRetag::No),
                            source_info,
                            location,
                        ))
                    } else {
                        op.clone()
                    }
                }
                Operand::Constant(_) | Operand::RuntimeChecks(_) => op.clone(),
            };
            assignments.push((dest_field, emit_op));
        }

        // Perform assignments *after* all aliasing fields have been read into
        // temporary locals.
        for (dest_field, emit_op) in assignments {
            self.patcher.add_assign(location, dest_field, Rvalue::Use(emit_op, WithRetag::No));
        }

        // Delete the original aggregate assignment.
        self.patcher.nop_statement(location);

        // For enum variants, set the discriminant after all field writes.
        if let Some(variant_index) = enum_variant {
            self.patcher.add_statement(
                location,
                StatementKind::SetDiscriminant { place: Box::new(dest), variant_index },
            );
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for AliasFixup<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        // Fixup the MIR to remove aliasing assignments.
        if let StatementKind::Assign((dest, rvalue)) = &mut statement.kind {
            match *rvalue {
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), with_retag) => {
                    if places_alias(self.tcx, self.local_decls, *dest, src) {
                        if src == *dest {
                            debug!("{:?} turned into self-assignment, deleting", location);
                            statement.make_nop(true);
                        } else {
                            let temp = self.isolate_rvalue_to_local(
                                rvalue.clone(),
                                statement.source_info,
                                location,
                            );
                            *rvalue = Rvalue::Use(Operand::Move(temp), with_retag);
                        }
                    }
                }
                Rvalue::Aggregate(AggregateKind::Array(_), ref mut operands) => self
                    .visit_aggregate_assign(
                        *dest,
                        None,
                        |tcx, place, field_idx, _field_ty| {
                            place.project_deeper(
                                &[PlaceElem::ConstantIndex {
                                    offset: field_idx.as_u32().into(),
                                    min_length: field_idx.as_u32() as u64 + 1,
                                    from_end: false,
                                }],
                                tcx,
                            )
                        },
                        operands,
                        statement.source_info,
                        location,
                    ),
                Rvalue::Aggregate(AggregateKind::Tuple, ref mut operands) => self
                    .visit_aggregate_assign(
                        *dest,
                        None,
                        |tcx, place, field_idx, field_ty| {
                            place.project_deeper(&[PlaceElem::Field(field_idx, field_ty)], tcx)
                        },
                        operands,
                        statement.source_info,
                        location,
                    ),
                Rvalue::Aggregate(
                    AggregateKind::Adt(_, _, _, _, Some(union_field_idx)),
                    ref mut operands,
                ) => {
                    debug_assert_eq!(operands.len(), 1);
                    self.visit_aggregate_assign(
                        *dest,
                        None,
                        |tcx, place, _, field_ty| {
                            place
                                .project_deeper(&[PlaceElem::Field(union_field_idx, field_ty)], tcx)
                        },
                        operands,
                        statement.source_info,
                        location,
                    )
                }
                Rvalue::Aggregate(
                    AggregateKind::Adt(adt_did, var_idx, _, _, None),
                    ref mut operands,
                ) => {
                    let def = self.tcx.adt_def(adt_did);
                    if def.repr().simd() {
                        // MCP#838 banned projections into SIMD types.
                        return;
                    }
                    self.visit_aggregate_assign(
                        *dest,
                        def.is_enum().then_some(var_idx),
                        |tcx, place, field_idx, field_ty| {
                            if def.is_enum() {
                                place.project_deeper(
                                    &[
                                        PlaceElem::Downcast(None, var_idx),
                                        PlaceElem::Field(field_idx, field_ty),
                                    ],
                                    tcx,
                                )
                            } else {
                                place.project_deeper(&[PlaceElem::Field(field_idx, field_ty)], tcx)
                            }
                        },
                        operands,
                        statement.source_info,
                        location,
                    )
                }

                // For other rvalues, don't try to split them into components
                // and instead just introduce a temporary if there is any
                // aliasing
                Rvalue::Aggregate(..)
                | Rvalue::Repeat(..)
                | Rvalue::Cast(..)
                | Rvalue::CopyForDeref(..)
                | Rvalue::WrapUnsafeBinder(..) => {
                    let mut overlaps_dest = false;
                    VisitPlacesWith(|place, _ctxt| {
                        if places_alias(self.tcx, self.local_decls, *dest, place) {
                            overlaps_dest = true;
                        }
                    })
                    .visit_rvalue(rvalue, location);
                    if overlaps_dest {
                        let temp = self.isolate_rvalue_to_local(
                            rvalue.clone(),
                            statement.source_info,
                            location,
                        );
                        *rvalue = Rvalue::Use(Operand::Move(temp), WithRetag::No);
                    }
                }

                // These permit either cannot have aliasing, or allow it because
                // they only operate on scalar backend types.
                Rvalue::Use(Operand::Constant(..) | Operand::RuntimeChecks(..), _)
                | Rvalue::Ref(..)
                | Rvalue::ThreadLocalRef(..)
                | Rvalue::BinaryOp(..)
                | Rvalue::UnaryOp(..)
                | Rvalue::Discriminant(..)
                | Rvalue::RawPtr(..)
                | Rvalue::Reborrow(..) => {}
            }
        }
    }
}
