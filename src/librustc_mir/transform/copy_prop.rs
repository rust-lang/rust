// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Trivial copy propagation pass.
//!
//! This uses def-use analysis to remove values that have exactly one def and one use, which must
//! be an assignment.
//!
//! To give an example, we look for patterns that look like:
//!
//!     DEST = SRC
//!     ...
//!     USE(DEST)
//!
//! where `DEST` and `SRC` are both locals of some form. We replace that with:
//!
//!     NOP
//!     ...
//!     USE(SRC)
//!
//! The assignment `DEST = SRC` must be (a) the only mutation of `DEST` and (b) the only
//! (non-mutating) use of `SRC`. These restrictions are conservative and may be relaxed in the
//! future.

use def_use::DefUseAnalysis;
use rustc::mir::repr::{Local, Lvalue, Mir, Operand, Rvalue, StatementKind};
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::ty::TyCtxt;
use rustc_data_structures::indexed_vec::Idx;

pub struct CopyPropagation;

impl Pass for CopyPropagation {}

impl<'tcx> MirPass<'tcx> for CopyPropagation {
    fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        loop {
            let mut def_use_analysis = DefUseAnalysis::new(mir);
            def_use_analysis.analyze(mir);

            let mut changed = false;
            for dest_local_index in 0..mir.count_locals() {
                let dest_local = Local::new(dest_local_index);
                debug!("Considering destination local: {}", mir.format_local(dest_local));

                let src_local;
                let location;
                {
                    // The destination must have exactly one def.
                    let dest_use_info = def_use_analysis.local_info(dest_local);
                    let dest_def_count = dest_use_info.def_count_not_including_drop();
                    if dest_def_count == 0 {
                        debug!("  Can't copy-propagate local: dest {} undefined",
                               mir.format_local(dest_local));
                        continue
                    }
                    if dest_def_count > 1 {
                        debug!("  Can't copy-propagate local: dest {} defined {} times",
                               mir.format_local(dest_local),
                               dest_use_info.def_count());
                        continue
                    }
                    if dest_use_info.use_count() == 0 {
                        debug!("  Can't copy-propagate local: dest {} unused",
                               mir.format_local(dest_local));
                        continue
                    }
                    let dest_lvalue_def = dest_use_info.defs_and_uses.iter().filter(|lvalue_def| {
                        lvalue_def.context.is_mutating_use() && !lvalue_def.context.is_drop()
                    }).next().unwrap();
                    location = dest_lvalue_def.location;

                    let basic_block = &mir[location.block];
                    let statement_index = location.statement_index;
                    let statement = match basic_block.statements.get(statement_index) {
                        Some(statement) => statement,
                        None => {
                            debug!("  Can't copy-propagate local: used in terminator");
                            continue
                        }
                    };

                    // That use of the source must be an assignment.
                    let src_lvalue = match statement.kind {
                        StatementKind::Assign(
                                ref dest_lvalue,
                                Rvalue::Use(Operand::Consume(ref src_lvalue)))
                                if Some(dest_local) == mir.local_index(dest_lvalue) => {
                            src_lvalue
                        }
                        _ => {
                            debug!("  Can't copy-propagate local: source use is not an \
                                    assignment");
                            continue
                        }
                    };
                    src_local = match mir.local_index(src_lvalue) {
                        Some(src_local) => src_local,
                        None => {
                            debug!("  Can't copy-propagate local: source is not a local");
                            continue
                        }
                    };

                    // There must be exactly one use of the source used in a statement (not in a
                    // terminator).
                    let src_use_info = def_use_analysis.local_info(src_local);
                    let src_use_count = src_use_info.use_count();
                    if src_use_count == 0 {
                        debug!("  Can't copy-propagate local: no uses");
                        continue
                    }
                    if src_use_count != 1 {
                        debug!("  Can't copy-propagate local: {} uses", src_use_info.use_count());
                        continue
                    }

                    // Verify that the source doesn't change in between. This is done
                    // conservatively for now, by ensuring that the source has exactly one
                    // mutation. The goal is to prevent things like:
                    //
                    //     DEST = SRC;
                    //     SRC = X;
                    //     USE(DEST);
                    //
                    // From being misoptimized into:
                    //
                    //     SRC = X;
                    //     USE(SRC);
                    let src_def_count = src_use_info.def_count_not_including_drop();
                    if src_def_count != 1 {
                        debug!("  Can't copy-propagate local: {} defs of src",
                               src_use_info.def_count_not_including_drop());
                        continue
                    }
                }

                // If all checks passed, then we can eliminate the destination and the assignment.
                //
                // First, remove all markers.
                //
                // FIXME(pcwalton): Don't do this. Merge live ranges instead.
                debug!("  Replacing all uses of {}", mir.format_local(dest_local));
                for lvalue_use in &def_use_analysis.local_info(dest_local).defs_and_uses {
                    if lvalue_use.context.is_storage_marker() {
                        mir.make_statement_nop(lvalue_use.location)
                    }
                }
                for lvalue_use in &def_use_analysis.local_info(src_local).defs_and_uses {
                    if lvalue_use.context.is_storage_marker() {
                        mir.make_statement_nop(lvalue_use.location)
                    }
                }

                // Now replace all uses of the destination local with the source local.
                let src_lvalue = Lvalue::from_local(mir, src_local);
                def_use_analysis.replace_all_defs_and_uses_with(dest_local, mir, src_lvalue);

                // Finally, zap the now-useless assignment instruction.
                mir.make_statement_nop(location);

                changed = true;
                // FIXME(pcwalton): Update the use-def chains to delete the instructions instead of
                // regenerating the chains.
                break
            }
            if !changed {
                break
            }
        }
    }
}

