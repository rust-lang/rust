// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that reuses "final destinations" of values,
//! propagating the lvalue back through a chain of moves.

use rustc::hir;
use rustc::mir::*;
use rustc::mir::visit::{LvalueContext, MutVisitor, Visitor};
use rustc::session::config::FullDebugInfo;
use rustc::ty::TyCtxt;
use transform::{MirPass, MirSource};

use rustc_data_structures::indexed_vec::IndexVec;

pub struct ReuseLvalues;

impl MirPass for ReuseLvalues {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Don't run on constant MIR, because trans might not be able to
        // evaluate the modified MIR.
        // FIXME(eddyb) Remove check after miri is merged.
        let id = tcx.hir.as_local_node_id(source.def_id).unwrap();
        match (tcx.hir.body_owner_kind(id), source.promoted) {
            (_, Some(_)) |
            (hir::BodyOwnerKind::Const, _) |
            (hir::BodyOwnerKind::Static(_), _) => return,

            (hir::BodyOwnerKind::Fn, _) => {
                if tcx.is_const_fn(source.def_id) {
                    // Don't run on const functions, as, again, trans might not be able to evaluate
                    // the optimized IR.
                    return
                }
            }
        }

        // FIXME(eddyb) We should allow multiple user variables
        // per local for debuginfo instead of not optimizing.
        if tcx.sess.opts.debuginfo == FullDebugInfo {
            return;
        }

        // Collect the def and move information for all locals.
        let mut collector = DefMoveCollector {
            defs_moves: IndexVec::from_elem((Def::Never, Move::Never), &mir.local_decls),
        };
        for arg in mir.args_iter() {
            // Arguments are special because they're initialized
            // in callers, and the collector doesn't know this.
            collector.defs_moves[arg].0 = Def::Other;
        }
        collector.visit_mir(mir);
        let mut defs_moves = collector.defs_moves;

        // Keep only locals with a clear initialization and move,
        // as they are guaranteed to have all accesses in between.
        // Also, the destination local of the move has to also have
        // a single initialization (the move itself), otherwise
        // there could be accesses that overlap the move chain.
        for local in mir.local_decls.indices() {
            let (def, mov) = defs_moves[local];
            if let Move::Into(dest) = mov {
                if let Def::Once { .. } = def {
                    if let (Def::Once { ref mut reused }, _) = defs_moves[dest] {
                        *reused = true;
                        continue;
                    }
                }

                // Failed to confirm the destination.
                defs_moves[local].1 = Move::Other;
            }
        }

        // Apply the appropriate replacements.
        let mut replacer = Replacer {
            defs_moves
        };
        replacer.visit_mir(mir);
    }
}

/// How (many times) was a local written?
/// Note that borrows are completely ignored,
/// as they get invalidated by moves regardless.
#[derive(Copy, Clone, Debug)]
enum Def {
    /// No writes to this local.
    Never,

    /// Only one direct initialization, from `Assign` or `Call`.
    ///
    /// If `reused` is `true`, this local is a destination
    /// in a chain of moves and should have all of its
    /// `StorageLive` statements removed.
    Once {
        reused: bool
    },

    /// Any other number or kind of mutating accesses.
    Other,
}

/// How (many times) was a local moved?
#[derive(Copy, Clone, Debug)]
enum Move {
    /// No moves of this local.
    Never,

    /// Only one move, assigning another local.
    ///
    /// Ends up replaced by its destination and should
    /// have all of its `StorageDead` statements removed.
    Into(Local),

    /// Any other number or kind of moves.
    Other,
}


struct DefMoveCollector {
    defs_moves: IndexVec<Local, (Def, Move)>,
}

impl<'tcx> Visitor<'tcx> for DefMoveCollector {
    fn visit_local(&mut self,
                   &local: &Local,
                   context: LvalueContext<'tcx>,
                   _location: Location) {
        let (ref mut def, ref mut mov) = self.defs_moves[local];
        match context {
            // We specifically want the first direct initialization.
            LvalueContext::Store |
            LvalueContext::Call => {
                if let Def::Never = *def {
                    *def = Def::Once { reused: false };
                } else {
                    *def = Def::Other;
                }
            }

            // Initialization of a field or similar.
            LvalueContext::Projection(Mutability::Mut) => {
                *def = Def::Other;
            }

            // Both of these count as moved, and not the kind
            // we want for `Move::Into` (see `visit_assign`).
            LvalueContext::Drop |
            LvalueContext::Move => *mov = Move::Other,

            // We can ignore everything else.
            LvalueContext::Inspect |
            LvalueContext::Copy |
            LvalueContext::Projection(Mutability::Not) |
            LvalueContext::Borrow { .. } |
            LvalueContext::StorageLive |
            LvalueContext::StorageDead |
            LvalueContext::Validate => {}
        }
    }

    fn visit_projection(&mut self,
                        proj: &LvalueProjection<'tcx>,
                        context: LvalueContext<'tcx>,
                        location: Location) {
        // Pretend derefs copy the underlying pointer, as we don't
        // need to treat the base local as being mutated or moved.
        let context = if let ProjectionElem::Deref = proj.elem {
            LvalueContext::Copy
        } else {
            match context {
                // Pass down the kinds of contexts for which we don't
                // need to disambigutate between direct and projected.
                LvalueContext::Borrow { .. } |
                LvalueContext::Copy |
                LvalueContext::Move |
                LvalueContext::Drop => context,

                _ if context.is_mutating_use() => {
                    LvalueContext::Projection(Mutability::Mut)
                }
                _ => {
                    LvalueContext::Projection(Mutability::Not)
                }
            }
        };
        self.visit_lvalue(&proj.base, context, location);
        self.visit_projection_elem(&proj.elem, context, location);
    }

    fn visit_assign(&mut self,
                    _: BasicBlock,
                    lvalue: &Lvalue<'tcx>,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        self.visit_lvalue(lvalue, LvalueContext::Store, location);

        // Handle `dest = move src`, and skip the `visit_local`
        // for `src`, which would always set it to `Move::Other`.
        match (lvalue, rvalue) {
            (&Lvalue::Local(dest), &Rvalue::Use(Operand::Move(Lvalue::Local(src)))) => {
                let (_, ref mut mov) = self.defs_moves[src];
                // We specifically want the first whole move into another local.
                if let Move::Never = *mov {
                    *mov = Move::Into(dest);
                } else {
                    *mov = Move::Other;
                }
            }
            _ => {
                self.visit_rvalue(rvalue, location);
            }
        }
    }
}

struct Replacer {
    defs_moves: IndexVec<Local, (Def, Move)>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for Replacer {
    fn visit_local(&mut self,
                   local: &mut Local,
                   context: LvalueContext<'tcx>,
                   location: Location) {
        let original = *local;
        if let (_, Move::Into(dest)) = self.defs_moves[original] {
            *local = dest;

            // Keep going, in case the move chain doesn't stop here.
            self.visit_local(local, context, location);

            // Cache the final result, in a similar way to union-find.
            let final_dest = *local;
            if final_dest != dest {
                self.defs_moves[original].1 = Move::Into(final_dest);
            }
        }
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        // Fuse storage liveness ranges of move chains, by removing
        // `StorageLive` of destinations and `StorageDead` of sources.
        match statement.kind {
            StatementKind::StorageLive(local) |
            StatementKind::StorageDead(local) => {
                // FIXME(eddyb) We also have to remove `StorageLive` of
                // sources and `StorageDead` of destinations to avoid
                // creating invalid storage liveness (half-)ranges.
                // The proper solution might involve recomputing them.
                match self.defs_moves[local] {
                    (Def::Once { reused: true }, _) |
                    (_, Move::Into(_)) => {
                        statement.kind = StatementKind::Nop;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        self.super_statement(block, statement, location);

        // Remove self-assignments resulting from replaced move chains.
        match statement.kind {
            StatementKind::Assign(Lvalue::Local(dest),
                Rvalue::Use(Operand::Move(Lvalue::Local(src)))) if dest == src => {
                    statement.kind = StatementKind::Nop;
                }
            _ => {}
        }
    }
}
