// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass adds validation calls (AcquireValid, ReleaseValid) where appropriate.
//! It has to be run really early, before transformations like inlining, because
//! introducing these calls *adds* UB -- so, conceptually, this pass is actually part
//! of MIR building, and only after this pass we think of the program has having the
//! normal MIR semantics.

use rustc::ty::{self, Ty, TyCtxt};
use rustc::mir::*;
use transform::{MirPass, MirSource};

pub struct AddRetag;

/// Determines whether this place is local: If it is part of a local variable.
/// We do not consider writes to pointers local, only writes that immediately assign
/// to a local variable.
/// One important property here is that evaluating the place immediately after
/// the assignment must produce the same place as what was used during the assignment.
fn is_local<'tcx>(
    place: &Place<'tcx>,
) -> bool {
    use rustc::mir::Place::*;

    match *place {
        Local { .. } => true,
        Promoted(_) |
        Static(_) => false,
        Projection(ref proj) => {
            match proj.elem {
                ProjectionElem::Deref |
                ProjectionElem::Index(_) =>
                    // Which place these point to depends on external circumstances
                    // (a local storing the array index, the current value of
                    // the projection base), so we stop tracking here.
                    false,
                ProjectionElem::Field { .. } |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Subslice { .. } |
                ProjectionElem::Downcast { .. } =>
                    // These just offset by a constant, entirely independent of everything else.
                    is_local(&proj.base),
            }
        }
    }
}

/// Determine whether this type has a reference in it, recursing below compound types but
/// not below references.
fn has_reference<'a, 'gcx, 'tcx>(ty: Ty<'tcx>, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> bool {
    match ty.sty {
        // Primitive types that are not references
        ty::Bool | ty::Char |
        ty::Float(_) | ty::Int(_) | ty::Uint(_) |
        ty::RawPtr(..) | ty::FnPtr(..) |
        ty::Str | ty::FnDef(..) | ty::Never =>
            false,
        // References
        ty::Ref(..) => true,
        ty::Adt(..) if ty.is_box() => true,
        // Compound types
        ty::Array(ty, ..) | ty::Slice(ty) =>
            has_reference(ty, tcx),
        ty::Tuple(tys) =>
            tys.iter().any(|ty| has_reference(ty, tcx)),
        ty::Adt(adt, substs) =>
            adt.variants.iter().any(|v| v.fields.iter().any(|f|
                has_reference(f.ty(tcx, substs), tcx)
            )),
        // Conservative fallback
        _ => true,
    }
}

impl MirPass for AddRetag {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _src: MirSource,
                          mir: &mut Mir<'tcx>)
    {
        if !tcx.sess.opts.debugging_opts.mir_emit_retag {
            return;
        }
        let (span, arg_count) = (mir.span, mir.arg_count);
        let (basic_blocks, local_decls) = mir.basic_blocks_and_local_decls_mut();
        let needs_retag = |place: &Place<'tcx>| {
            is_local(place) && has_reference(place.ty(&*local_decls, tcx).to_ty(tcx), tcx)
        };

        // PART 1
        // Retag arguments at the beginning of the start block.
        {
            let source_info = SourceInfo {
                scope: OUTERMOST_SOURCE_SCOPE,
                span: span, // FIXME: Consider using just the span covering the function
                            // argument declaration.
            };
            // Gather all arguments, skip return value.
            let places = local_decls.iter_enumerated().skip(1).take(arg_count)
                    .map(|(local, _)| Place::Local(local))
                    .filter(needs_retag)
                    .collect::<Vec<_>>();
            // Emit their retags.
            basic_blocks[START_BLOCK].statements.splice(0..0,
                places.into_iter().map(|place| Statement {
                    source_info,
                    kind: StatementKind::Retag { fn_entry: true, place },
                })
            );
        }

        // PART 2
        // Retag return values of functions.
        // We collect the return destinations because we cannot mutate while iterating.
        let mut returns: Vec<(SourceInfo, Place<'tcx>, BasicBlock)> = Vec::new();
        for block_data in basic_blocks.iter_mut() {
            match block_data.terminator {
                Some(Terminator { kind: TerminatorKind::Call { ref destination, .. },
                                  source_info }) => {
                    // Remember the return destination for later
                    if let Some(ref destination) = destination {
                        if needs_retag(&destination.0) {
                            returns.push((source_info, destination.0.clone(), destination.1));
                        }
                    }
                }
                _ => {
                    // Not a block ending in a Call -> ignore.
                    // `Drop` is also a call, but it doesn't return anything so we are good.
                }
            }
        }
        // Now we go over the returns we collected to retag the return values.
        for (source_info, dest_place, dest_block) in returns {
            basic_blocks[dest_block].statements.insert(0, Statement {
                source_info,
                kind: StatementKind::Retag { fn_entry: false, place: dest_place },
            });
        }

        // PART 3
        // Add retag after assignment.
        for block_data in basic_blocks {
            // We want to insert statements as we iterate.  To this end, we
            // iterate backwards using indices.
            for i in (0..block_data.statements.len()).rev() {
                match block_data.statements[i].kind {
                    // Assignments can make values obtained elsewhere "local".
                    // We could try to be smart here and e.g. only retag if the assignment
                    // loaded from memory, but that seems risky: We might miss a subtle corner
                    // case.
                    StatementKind::Assign(ref place, box Rvalue::Use(..))
                    if needs_retag(place) => {
                        // Insert a retag after the assignment.
                        let source_info = block_data.statements[i].source_info;
                        block_data.statements.insert(i+1,Statement {
                            source_info,
                            kind: StatementKind::Retag { fn_entry: false, place: place.clone() },
                        });
                    }
                    _ => {},
                }
            }
        }
    }
}
