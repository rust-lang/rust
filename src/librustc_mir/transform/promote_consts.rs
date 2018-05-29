// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that promotes borrows of constant rvalues.
//!
//! The rvalues considered constant are trees of temps,
//! each with exactly one initialization, and holding
//! a constant value with no interior mutability.
//! They are placed into a new MIR constant body in
//! `promoted` and the borrow rvalue is replaced with
//! a `Literal::Promoted` using the index into `promoted`
//! of that constant MIR.
//!
//! This pass assumes that every use is dominated by an
//! initialization and can otherwise silence errors, if
//! move analysis runs after promotion on broken MIR.

use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc::mir::traversal::ReversePostorder;
use rustc::ty::{self, TyCtxt};
use syntax_pos::Span;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use std::{cmp, iter, mem, usize};

/// State of a temporary during collection and promotion.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TempState {
    /// No references to this temp.
    Undefined,
    /// One direct assignment and any number of direct uses.
    /// A borrow of this temp is promotable if the assigned
    /// value is qualified as constant.
    Defined {
        location: Location,
        uses: usize
    },
    /// Any other combination of assignments/uses.
    Unpromotable,
    /// This temp was part of an rvalue which got extracted
    /// during promotion and needs cleanup.
    PromotedOut
}

impl TempState {
    pub fn is_promotable(&self) -> bool {
        if let TempState::Defined { uses, .. } = *self {
            uses > 0
        } else {
            false
        }
    }
}

/// A "root candidate" for promotion, which will become the
/// returned value in a promoted MIR, unless it's a subset
/// of a larger candidate.
#[derive(Debug)]
pub enum Candidate {
    /// Borrow of a constant temporary.
    Ref(Location),

    /// Currently applied to function calls where the callee has the unstable
    /// `#[rustc_args_required_const]` attribute as well as the SIMD shuffle
    /// intrinsic. The intrinsic requires the arguments are indeed constant and
    /// the attribute currently provides the semantic requirement that arguments
    /// must be constant.
    Argument { bb: BasicBlock, index: usize },
}

struct TempCollector<'tcx> {
    temps: IndexVec<Local, TempState>,
    span: Span,
    mir: &'tcx Mir<'tcx>,
}

impl<'tcx> Visitor<'tcx> for TempCollector<'tcx> {
    fn visit_local(&mut self,
                   &index: &Local,
                   context: PlaceContext<'tcx>,
                   location: Location) {
        // We're only interested in temporaries
        if self.mir.local_kind(index) != LocalKind::Temp {
            return;
        }

        // Ignore drops, if the temp gets promoted,
        // then it's constant and thus drop is noop.
        // Storage live ranges are also irrelevant.
        if context.is_drop() || context.is_storage_marker() {
            return;
        }

        let temp = &mut self.temps[index];
        if *temp == TempState::Undefined {
            match context {
                PlaceContext::Store |
                PlaceContext::AsmOutput |
                PlaceContext::Call => {
                    *temp = TempState::Defined {
                        location,
                        uses: 0
                    };
                    return;
                }
                _ => { /* mark as unpromotable below */ }
            }
        } else if let TempState::Defined { ref mut uses, .. } = *temp {
            // We always allow borrows, even mutable ones, as we need
            // to promote mutable borrows of some ZSTs e.g. `&mut []`.
            let allowed_use = match context {
                PlaceContext::Borrow {..} => true,
                _ => context.is_nonmutating_use()
            };
            if allowed_use {
                *uses += 1;
                return;
            }
            /* mark as unpromotable below */
        }
        *temp = TempState::Unpromotable;
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.span = source_info.span;
    }
}

pub fn collect_temps(mir: &Mir, rpo: &mut ReversePostorder) -> IndexVec<Local, TempState> {
    let mut collector = TempCollector {
        temps: IndexVec::from_elem(TempState::Undefined, &mir.local_decls),
        span: mir.span,
        mir,
    };
    for (bb, data) in rpo {
        collector.visit_basic_block_data(bb, data);
    }
    collector.temps
}

struct Promoter<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    source: &'a mut Mir<'tcx>,
    promoted: Mir<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,
    extra_statements: &'a mut Vec<(Location, Statement<'tcx>)>,

    /// If true, all nested temps are also kept in the
    /// source MIR, not moved to the promoted MIR.
    keep_original: bool
}

impl<'a, 'tcx> Promoter<'a, 'tcx> {
    fn new_block(&mut self) -> BasicBlock {
        let span = self.promoted.span;
        self.promoted.basic_blocks_mut().push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: SourceInfo {
                    span,
                    scope: OUTERMOST_SOURCE_SCOPE
                },
                kind: TerminatorKind::Return
            }),
            is_cleanup: false
        })
    }

    fn assign(&mut self, dest: Local, rvalue: Rvalue<'tcx>, span: Span) {
        let last = self.promoted.basic_blocks().last().unwrap();
        let data = &mut self.promoted[last];
        data.statements.push(Statement {
            source_info: SourceInfo {
                span,
                scope: OUTERMOST_SOURCE_SCOPE
            },
            kind: StatementKind::Assign(Place::Local(dest), rvalue)
        });
    }

    /// Copy the initialization of this temp to the
    /// promoted MIR, recursing through temps.
    fn promote_temp(&mut self, temp: Local) -> Local {
        let old_keep_original = self.keep_original;
        let loc = match self.temps[temp] {
            TempState::Defined { location, uses } if uses > 0 => {
                if uses > 1 {
                    self.keep_original = true;
                }
                location
            }
            state =>  {
                span_bug!(self.promoted.span, "{:?} not promotable: {:?}",
                          temp, state);
            }
        };
        if !self.keep_original {
            self.temps[temp] = TempState::PromotedOut;
        }

        let no_stmts = self.source[loc.block].statements.len();
        let new_temp = self.promoted.local_decls.push(
            LocalDecl::new_temp(self.source.local_decls[temp].ty,
                                self.source.local_decls[temp].source_info.span));

        debug!("promote({:?} @ {:?}/{:?}, {:?})",
               temp, loc, no_stmts, self.keep_original);

        // First, take the Rvalue or Call out of the source MIR,
        // or duplicate it, depending on keep_original.
        if loc.statement_index < no_stmts {
            let (mut rvalue, source_info) = {
                let statement = &mut self.source[loc.block].statements[loc.statement_index];
                let rhs = match statement.kind {
                    StatementKind::Assign(_, ref mut rhs) => rhs,
                    _ => {
                        span_bug!(statement.source_info.span, "{:?} is not an assignment",
                                  statement);
                    }
                };

                (if self.keep_original {
                    rhs.clone()
                } else {
                    let unit = Rvalue::Aggregate(box AggregateKind::Tuple, vec![]);
                    mem::replace(rhs, unit)
                }, statement.source_info)
            };

            self.visit_rvalue(&mut rvalue, loc);
            self.assign(new_temp, rvalue, source_info.span);
        } else {
            let terminator = if self.keep_original {
                self.source[loc.block].terminator().clone()
            } else {
                let terminator = self.source[loc.block].terminator_mut();
                let target = match terminator.kind {
                    TerminatorKind::Call { destination: Some((_, target)), .. } => target,
                    ref kind => {
                        span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                    }
                };
                Terminator {
                    source_info: terminator.source_info,
                    kind: mem::replace(&mut terminator.kind, TerminatorKind::Goto {
                        target,
                    })
                }
            };

            match terminator.kind {
                TerminatorKind::Call { mut func, mut args, .. } => {
                    self.visit_operand(&mut func, loc);
                    for arg in &mut args {
                        self.visit_operand(arg, loc);
                    }

                    let last = self.promoted.basic_blocks().last().unwrap();
                    let new_target = self.new_block();

                    *self.promoted[last].terminator_mut() = Terminator {
                        kind: TerminatorKind::Call {
                            func,
                            args,
                            cleanup: None,
                            destination: Some((Place::Local(new_temp), new_target))
                        },
                        ..terminator
                    };
                }
                ref kind => {
                    span_bug!(terminator.source_info.span, "{:?} not promotable", kind);
                }
            };
        };

        self.keep_original = old_keep_original;
        new_temp
    }

    fn promote_candidate(mut self, candidate: Candidate) {
        let mut rvalue = {
            let promoted = &mut self.promoted;
            let literal = Literal::Promoted {
                index: Promoted::new(self.source.promoted.len())
            };
            let operand = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] =
                    LocalDecl::new_return_place(ty, span);
                Operand::Constant(box Constant {
                    span,
                    ty,
                    literal
                })
            };
            let (blocks, local_decls) = self.source.basic_blocks_and_local_decls_mut();
            match candidate {
                Candidate::Ref(loc) => {
                    let ref mut statement = blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(_, Rvalue::Ref(r, bk, ref mut place)) => {
                            // Find the underlying local for this (necessarilly interior) borrow.
                            // HACK(eddyb) using a recursive function because of mutable borrows.
                            fn interior_base<'a, 'tcx>(place: &'a mut Place<'tcx>)
                                                       -> &'a mut Place<'tcx> {
                                if let Place::Projection(ref mut proj) = *place {
                                    assert_ne!(proj.elem, ProjectionElem::Deref);
                                    return interior_base(&mut proj.base);
                                }
                                place
                            }
                            let place = interior_base(place);

                            let ty = place.ty(local_decls, self.tcx).to_ty(self.tcx);
                            let ref_ty = self.tcx.mk_ref(r,
                                ty::TypeAndMut {
                                    ty,
                                    mutbl: bk.to_mutbl_lossy()
                                }
                            );
                            let span = statement.source_info.span;

                            // Create a temp to hold the promoted reference.
                            // This is because `*r` requires `r` to be a local,
                            // otherwise we would use the `promoted` directly.
                            let mut promoted_ref = LocalDecl::new_temp(ref_ty, span);
                            promoted_ref.source_info = statement.source_info;
                            promoted_ref.visibility_scope = statement.source_info.scope;
                            let promoted_ref = local_decls.push(promoted_ref);
                            assert_eq!(self.temps.push(TempState::Unpromotable), promoted_ref);
                            self.extra_statements.push((loc, Statement {
                                source_info: statement.source_info,
                                kind: StatementKind::Assign(
                                    Place::Local(promoted_ref),
                                    Rvalue::Use(operand(ref_ty, span)),
                                )
                            }));
                            let promoted_place = Place::Local(promoted_ref).deref();

                            Rvalue::Ref(r, bk, mem::replace(place, promoted_place))
                        }
                        _ => bug!()
                    }
                }
                Candidate::Argument { bb, index } => {
                    let terminator = blocks[bb].terminator_mut();
                    match terminator.kind {
                        TerminatorKind::Call { ref mut args, .. } => {
                            let ty = args[index].ty(local_decls, self.tcx);
                            let span = terminator.source_info.span;
                            Rvalue::Use(mem::replace(&mut args[index], operand(ty, span)))
                        }
                        _ => bug!()
                    }
                }
            }
        };

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_rvalue(&mut rvalue, Location {
            block: BasicBlock::new(0),
            statement_index: usize::MAX
        });

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, rvalue, span);
        self.source.promoted.push(self.promoted);
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        if self.source.local_kind(*local) == LocalKind::Temp {
            *local = self.promote_temp(*local);
        }
    }
}

pub fn promote_candidates<'a, 'tcx>(mir: &mut Mir<'tcx>,
                                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    mut temps: IndexVec<Local, TempState>,
                                    candidates: Vec<Candidate>) {
    // Visit candidates in reverse, in case they're nested.
    debug!("promote_candidates({:?})", candidates);

    let mut extra_statements = vec![];
    for candidate in candidates.into_iter().rev() {
        match candidate {
            Candidate::Ref(Location { block, statement_index }) => {
                match mir[block].statements[statement_index].kind {
                    StatementKind::Assign(Place::Local(local), _) => {
                        if temps[local] == TempState::PromotedOut {
                            // Already promoted.
                            continue;
                        }
                    }
                    _ => {}
                }
            }
            Candidate::Argument { .. } => {}
        }


        // Declare return place local so that `Mir::new` doesn't complain.
        let initial_locals = iter::once(
            LocalDecl::new_return_place(tcx.types.never, mir.span)
        ).collect();

        let mut promoter = Promoter {
            promoted: Mir::new(
                IndexVec::new(),
                // FIXME: maybe try to filter this to avoid blowing up
                // memory usage?
                mir.source_scopes.clone(),
                mir.source_scope_local_data.clone(),
                IndexVec::new(),
                None,
                initial_locals,
                0,
                vec![],
                mir.span
            ),
            tcx,
            source: mir,
            temps: &mut temps,
            extra_statements: &mut extra_statements,
            keep_original: false
        };
        promoter.promote_candidate(candidate);
    }

    // Insert each of `extra_statements` before its indicated location, which
    // has to be done in reverse location order, to not invalidate the rest.
    extra_statements.sort_by_key(|&(loc, _)| cmp::Reverse(loc));
    for (loc, statement) in extra_statements {
        mir[loc.block].statements.insert(loc.statement_index, statement);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in mir.basic_blocks_mut() {
        block.statements.retain(|statement| {
            match statement.kind {
                StatementKind::Assign(Place::Local(index), _) |
                StatementKind::StorageLive(index) |
                StatementKind::StorageDead(index) => {
                    !promoted(index)
                }
                _ => true
            }
        });
        let terminator = block.terminator_mut();
        match terminator.kind {
            TerminatorKind::Drop { location: Place::Local(index), target, .. } => {
                if promoted(index) {
                    terminator.kind = TerminatorKind::Goto {
                        target,
                    };
                }
            }
            _ => {}
        }
    }
}
