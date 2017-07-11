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

use rustc::ty::{TyCtxt, RegionKind};
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource};

pub struct AddValidation;

impl MirPass for AddValidation {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        // PART 1
        // Add an AcquireValid at the beginning of the start block.
        if mir.arg_count > 0 {
            let acquire_stmt = Statement {
                source_info: SourceInfo {
                    scope: ARGUMENT_VISIBILITY_SCOPE,
                    span: mir.span, // TODO: Consider using just the span covering the function argument declaration
                },
                kind: StatementKind::Validate(ValidationOp::Acquire,
                    // Skip return value, go over all the arguments
                    mir.local_decls.iter_enumerated().skip(1).take(mir.arg_count)
                    .map(|(local, local_decl)| (local_decl.ty, Lvalue::Local(local))).collect()
                )
            };
            mir.basic_blocks_mut()[START_BLOCK].statements.insert(0, acquire_stmt);
        }

        // PART 2
        // Add ReleaseValid/AcquireValid around function call terminators.  We don't use a visitor because
        // we need to access the block that a Call jumps to.
        let mut returns : Vec<(SourceInfo, Lvalue<'tcx>, BasicBlock)> = Vec::new(); // Here we collect the destinations.
        let local_decls = mir.local_decls.clone(); // TODO: Find a way to get rid of this clone.
        for block_data in mir.basic_blocks_mut() {
            match block_data.terminator {
                Some(Terminator { kind: TerminatorKind::Call { ref args, ref destination, .. }, source_info }) => {
                    // Before the call: Release all arguments
                    let release_stmt = Statement {
                        source_info,
                        kind: StatementKind::Validate(ValidationOp::Release,
                            args.iter().filter_map(|op| {
                                match op {
                                    &Operand::Consume(ref lval) => {
                                        let ty = lval.ty(&local_decls, tcx).to_ty(tcx);
                                        Some((ty, lval.clone()))
                                    },
                                    &Operand::Constant(..) => { None },
                                }
                            }).collect())
                    };
                    block_data.statements.push(release_stmt);
                    // Remember the return destination for later
                    if let &Some(ref destination) = destination {
                        returns.push((source_info, destination.0.clone(), destination.1));
                    }
                }
                _ => {
                    // Not a block ending in a Call -> ignore.
                    // TODO: Handle drop.
                }
            }
        }
        // Now we go over the returns we collected to acquire the return values.
        for (source_info, dest_lval, dest_block) in returns {
            let ty = dest_lval.ty(&local_decls, tcx).to_ty(tcx);
            let acquire_stmt = Statement {
                source_info,
                kind: StatementKind::Validate(ValidationOp::Acquire, vec![(ty, dest_lval)])
            };
            mir.basic_blocks_mut()[dest_block].statements.insert(0, acquire_stmt);
        }

        // PART 3
        // Add ReleaseValid/AcquireValid around Ref.  Again an iterator does not seem very suited as
        // we need to add new statements before and after each Ref.
        for block_data in mir.basic_blocks_mut() {
            // We want to insert statements around Ref commands as we iterate.  To this end, we iterate backwards
            // using indices.
            for i in (0..block_data.statements.len()).rev() {
                let (dest_lval, re, src_lval) = match block_data.statements[i].kind {
                    StatementKind::Assign(ref dest_lval, Rvalue::Ref(re, _, ref src_lval)) => {
                        (dest_lval.clone(), re, src_lval.clone())
                    },
                    _ => continue,
                };
                // So this is a ref, and we got all the data we wanted.
                let dest_ty = dest_lval.ty(&local_decls, tcx).to_ty(tcx);
                let acquire_stmt = Statement {
                    source_info: block_data.statements[i].source_info,
                    kind: StatementKind::Validate(ValidationOp::Acquire, vec![(dest_ty, dest_lval)]),
                };
                block_data.statements.insert(i+1, acquire_stmt);

                // The source is released until the region of the borrow ends.
                // FIXME: We have to check whether the source path was writable.
                let src_ty = src_lval.ty(&local_decls, tcx).to_ty(tcx);
                let op = match re {
                    &RegionKind::ReScope(ce) => ValidationOp::Suspend(ce),
                    &RegionKind::ReErased => bug!("AddValidation pass must be run before erasing lifetimes"),
                    _ => ValidationOp::Release,
                };
                let release_stmt = Statement {
                    source_info: block_data.statements[i].source_info,
                    kind: StatementKind::Validate(op, vec![(src_ty, src_lval)]),
                };
                block_data.statements.insert(i, release_stmt);
            }
        }
    }
}
