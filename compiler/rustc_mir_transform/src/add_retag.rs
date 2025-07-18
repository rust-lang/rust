//! This pass adds validation calls (AcquireValid, ReleaseValid) where appropriate.
//! It has to be run really early, before transformations like inlining, because
//! introducing these calls *adds* UB -- so, conceptually, this pass is actually part
//! of MIR building, and only after this pass we think of the program has having the
//! normal MIR semantics.

use rustc_hir::LangItem;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub(super) struct AddRetag;

/// Determine whether this type may contain a reference (or box), and thus needs retagging.
/// We will only recurse `depth` times into Tuples/ADTs to bound the cost of this.
fn may_contain_reference<'tcx>(ty: Ty<'tcx>, depth: u32, tcx: TyCtxt<'tcx>) -> bool {
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
        // References and Boxes (`noalias` sources)
        ty::Ref(..) => true,
        ty::Adt(..) if ty.is_box() => true,
        ty::Adt(adt, _) if tcx.is_lang_item(adt.did(), LangItem::PtrUnique) => true,
        // Compound types: recurse
        ty::Array(ty, _) | ty::Slice(ty) => {
            // This does not branch so we keep the depth the same.
            may_contain_reference(*ty, depth, tcx)
        }
        ty::Tuple(tys) => {
            depth == 0 || tys.iter().any(|ty| may_contain_reference(ty, depth - 1, tcx))
        }
        ty::Adt(adt, args) => {
            depth == 0
                || adt.variants().iter().any(|v| {
                    v.fields.iter().any(|f| may_contain_reference(f.ty(tcx, args), depth - 1, tcx))
                })
        }
        // Conservative fallback
        _ => true,
    }
}

impl<'tcx> crate::MirPass<'tcx> for AddRetag {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.mir_emit_retag
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // We need an `AllCallEdges` pass before we can do any work.
        super::add_call_guards::AllCallEdges.run_pass(tcx, body);

        let basic_blocks = body.basic_blocks.as_mut();
        let local_decls = &body.local_decls;
        let needs_retag = |place: &Place<'tcx>| {
            // We're not really interested in stores to "outside" locations, they are hard to keep
            // track of anyway.
            !place.is_indirect_first_projection()
                && may_contain_reference(place.ty(&*local_decls, tcx).ty, /*depth*/ 3, tcx)
                && !local_decls[place.local].is_deref_temp()
        };

        // PART 1
        // Retag arguments at the beginning of the start block.
        {
            // Gather all arguments, skip return value.
            let places = local_decls.iter_enumerated().skip(1).take(body.arg_count).filter_map(
                |(local, decl)| {
                    let place = Place::from(local);
                    needs_retag(&place).then_some((place, decl.source_info))
                },
            );

            // Emit their retags.
            basic_blocks[START_BLOCK].statements.splice(
                0..0,
                places.map(|(place, source_info)| {
                    Statement::new(
                        source_info,
                        StatementKind::Retag(RetagKind::FnEntry, Box::new(place)),
                    )
                }),
            );
        }

        // PART 2
        // Retag return values of functions.
        // We collect the return destinations because we cannot mutate while iterating.
        let returns = basic_blocks
            .iter_mut()
            .filter_map(|block_data| {
                match block_data.terminator().kind {
                    TerminatorKind::Call { target: Some(target), destination, .. }
                        if needs_retag(&destination) =>
                    {
                        // Remember the return destination for later
                        Some((block_data.terminator().source_info, destination, target))
                    }

                    // `Drop` is also a call, but it doesn't return anything so we are good.
                    TerminatorKind::Drop { .. } => None,
                    // Not a block ending in a Call -> ignore.
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        // Now we go over the returns we collected to retag the return values.
        for (source_info, dest_place, dest_block) in returns {
            basic_blocks[dest_block].statements.insert(
                0,
                Statement::new(
                    source_info,
                    StatementKind::Retag(RetagKind::Default, Box::new(dest_place)),
                ),
            );
        }

        // PART 3
        // Add retag after assignments.
        for block_data in basic_blocks {
            // We want to insert statements as we iterate. To this end, we
            // iterate backwards using indices.
            for i in (0..block_data.statements.len()).rev() {
                let (retag_kind, place) = match block_data.statements[i].kind {
                    // Retag after assignments of reference type.
                    StatementKind::Assign(box (ref place, ref rvalue)) => {
                        let add_retag = match rvalue {
                            // Ptr-creating operations already do their own internal retagging, no
                            // need to also add a retag statement. *Except* if we are deref'ing a
                            // Box, because those get desugared to directly working with the inner
                            // raw pointer! That's relevant for `RawPtr` as Miri otherwise makes it
                            // a NOP when the original pointer is already raw.
                            Rvalue::RawPtr(_mutbl, place) => {
                                // Using `is_box_global` here is a bit sketchy: if this code is
                                // generic over the allocator, we'll not add a retag! This is a hack
                                // to make Stacked Borrows compatible with custom allocator code.
                                // It means the raw pointer inherits the tag of the box, which mostly works
                                // but can sometimes lead to unexpected aliasing errors.
                                // Long-term, we'll want to move to an aliasing model where "cast to
                                // raw pointer" is a complete NOP, and then this will no longer be
                                // an issue.
                                if place.is_indirect_first_projection()
                                    && body.local_decls[place.local].ty.is_box_global(tcx)
                                {
                                    Some(RetagKind::Raw)
                                } else {
                                    None
                                }
                            }
                            Rvalue::Ref(..) => None,
                            _ => {
                                if needs_retag(place) {
                                    Some(RetagKind::Default)
                                } else {
                                    None
                                }
                            }
                        };
                        if let Some(kind) = add_retag {
                            (kind, *place)
                        } else {
                            continue;
                        }
                    }
                    // Do nothing for the rest
                    _ => continue,
                };
                // Insert a retag after the statement.
                let source_info = block_data.statements[i].source_info;
                block_data.statements.insert(
                    i + 1,
                    Statement::new(source_info, StatementKind::Retag(retag_kind, Box::new(place))),
                );
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
