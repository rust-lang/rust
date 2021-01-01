//! This pass adds validation calls (AcquireValid, ReleaseValid) where appropriate.
//! It has to be run really early, before transformations like inlining, because
//! introducing these calls *adds* UB -- so, conceptually, this pass is actually part
//! of MIR building, and only after this pass we think of the program has having the
//! normal MIR semantics.

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct AddRetag;

/// Determines whether this place is "stable": Whether, if we evaluate it again
/// after the assignment, we can be sure to obtain the same place value.
/// (Concurrent accesses by other threads are no problem as these are anyway non-atomic
/// copies.  Data races are UB.)
fn is_stable(place: PlaceRef<'_>) -> bool {
    place.projection.iter().all(|elem| {
        match elem {
            // Which place this evaluates to can change with any memory write,
            // so cannot assume this to be stable.
            ProjectionElem::Deref => false,
            // Array indices are interesting, but MIR building generates a *fresh*
            // temporary for every array access, so the index cannot be changed as
            // a side-effect.
            ProjectionElem::Index { .. } |
            // The rest is completely boring, they just offset by a constant.
            ProjectionElem::Field { .. } |
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Subslice { .. } |
            ProjectionElem::Downcast { .. } => true,
        }
    })
}

/// Determine whether this type may be a reference (or box), and thus needs retagging.
fn may_be_reference(ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // Primitive types that are not references
        ty::Bool
        | ty::Char
        | ty::Float(_)
        | ty::Int(_)
        | ty::Uint(_)
        | ty::RawPtr(..)
        | ty::FnPtr(..)
        | ty::Str
        | ty::FnDef(..)
        | ty::Never => false,
        // References
        ty::Ref(..) => true,
        ty::Adt(..) if ty.is_box() => true,
        // Compound types are not references
        ty::Array(..) | ty::Slice(..) | ty::Tuple(..) | ty::Adt(..) => false,
        // Conservative fallback
        _ => true,
    }
}

impl<'tcx> MirPass<'tcx> for AddRetag {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if !tcx.sess.opts.debugging_opts.mir_emit_retag {
            return;
        }

        // We need an `AllCallEdges` pass before we can do any work.
        super::add_call_guards::AllCallEdges.run_pass(tcx, body);

        let (span, arg_count) = (body.span, body.arg_count);
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        let needs_retag = |place: &Place<'tcx>| {
            // FIXME: Instead of giving up for unstable places, we should introduce
            // a temporary and retag on that.
            is_stable(place.as_ref()) && may_be_reference(place.ty(&*local_decls, tcx).ty)
        };
        let place_base_raw = |place: &Place<'tcx>| {
            // If this is a `Deref`, get the type of what we are deref'ing.
            let deref_base =
                place.projection.iter().rposition(|p| matches!(p, ProjectionElem::Deref));
            if let Some(deref_base) = deref_base {
                let base_proj = &place.projection[..deref_base];
                let ty = Place::ty_from(place.local, base_proj, &*local_decls, tcx).ty;
                ty.is_unsafe_ptr()
            } else {
                // Not a deref, and thus not raw.
                false
            }
        };

        // PART 1
        // Retag arguments at the beginning of the start block.
        {
            // FIXME: Consider using just the span covering the function
            // argument declaration.
            let source_info = SourceInfo::outermost(span);
            // Gather all arguments, skip return value.
            let places = local_decls
                .iter_enumerated()
                .skip(1)
                .take(arg_count)
                .map(|(local, _)| Place::from(local))
                .filter(needs_retag);
            // Emit their retags.
            basic_blocks[START_BLOCK].statements.splice(
                0..0,
                places.map(|place| Statement {
                    source_info,
                    kind: StatementKind::Retag(RetagKind::FnEntry, Box::new(place)),
                }),
            );
        }

        // PART 2
        // Retag return values of functions.  Also escape-to-raw the argument of `drop`.
        // We collect the return destinations because we cannot mutate while iterating.
        let returns = basic_blocks
            .iter_mut()
            .filter_map(|block_data| {
                match block_data.terminator().kind {
                    TerminatorKind::Call { destination: Some(ref destination), .. }
                        if needs_retag(&destination.0) =>
                    {
                        // Remember the return destination for later
                        Some((block_data.terminator().source_info, destination.0, destination.1))
                    }

                    // `Drop` is also a call, but it doesn't return anything so we are good.
                    TerminatorKind::Drop { .. } | TerminatorKind::DropAndReplace { .. } => None,
                    // Not a block ending in a Call -> ignore.
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        // Now we go over the returns we collected to retag the return values.
        for (source_info, dest_place, dest_block) in returns {
            basic_blocks[dest_block].statements.insert(
                0,
                Statement {
                    source_info,
                    kind: StatementKind::Retag(RetagKind::Default, Box::new(dest_place)),
                },
            );
        }

        // PART 3
        // Add retag after assignment.
        for block_data in basic_blocks {
            // We want to insert statements as we iterate.  To this end, we
            // iterate backwards using indices.
            for i in (0..block_data.statements.len()).rev() {
                let (retag_kind, place) = match block_data.statements[i].kind {
                    // Retag-as-raw after escaping to a raw pointer, if the referent
                    // is not already a raw pointer.
                    StatementKind::Assign(box (lplace, Rvalue::AddressOf(_, ref rplace)))
                        if !place_base_raw(rplace) =>
                    {
                        (RetagKind::Raw, lplace)
                    }
                    // Retag after assignments of reference type.
                    StatementKind::Assign(box (ref place, ref rvalue)) if needs_retag(place) => {
                        let kind = match rvalue {
                            Rvalue::Ref(_, borrow_kind, _)
                                if borrow_kind.allows_two_phase_borrow() =>
                            {
                                RetagKind::TwoPhase
                            }
                            _ => RetagKind::Default,
                        };
                        (kind, *place)
                    }
                    // Do nothing for the rest
                    _ => continue,
                };
                // Insert a retag after the statement.
                let source_info = block_data.statements[i].source_info;
                block_data.statements.insert(
                    i + 1,
                    Statement {
                        source_info,
                        kind: StatementKind::Retag(retag_kind, Box::new(place)),
                    },
                );
            }
        }
    }
}
