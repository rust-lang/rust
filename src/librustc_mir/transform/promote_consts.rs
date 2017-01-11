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
use rustc::mir::visit::{LvalueContext, MutVisitor, Visitor};
use rustc::mir::traversal::ReversePostorder;
use rustc::ty::TyCtxt;
use syntax_pos::Span;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use std::iter;
use std::mem;
use std::usize;

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

    /// Array of indices found in the third argument of
    /// a call to one of the simd_shuffleN intrinsics.
    ShuffleIndices(BasicBlock)
}

struct TempCollector<'tcx> {
    temps: IndexVec<Local, TempState>,
    span: Span,
    mir: &'tcx Mir<'tcx>,
}

impl<'tcx> Visitor<'tcx> for TempCollector<'tcx> {
    fn visit_lvalue(&mut self,
                    lvalue: &Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        self.super_lvalue(lvalue, context, location);
        if let Lvalue::Local(index) = *lvalue {
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
                    LvalueContext::Store |
                    LvalueContext::Call => {
                        *temp = TempState::Defined {
                            location: location,
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
                    LvalueContext::Borrow {..} => true,
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
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        self.span = source_info.span;
    }
}

pub fn collect_temps(mir: &Mir, rpo: &mut ReversePostorder) -> IndexVec<Local, TempState> {
    let mut collector = TempCollector {
        temps: IndexVec::from_elem(TempState::Undefined, &mir.local_decls),
        span: mir.span,
        mir: mir,
    };
    for (bb, data) in rpo {
        collector.visit_basic_block_data(bb, data);
    }
    collector.temps
}

struct Promoter<'a, 'tcx: 'a> {
    source: &'a mut Mir<'tcx>,
    promoted: Mir<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,

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
                    span: span,
                    scope: ARGUMENT_VISIBILITY_SCOPE
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
                span: span,
                scope: ARGUMENT_VISIBILITY_SCOPE
            },
            kind: StatementKind::Assign(Lvalue::Local(dest), rvalue)
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
            LocalDecl::new_temp(self.source.local_decls[temp].ty));

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
                    let unit = Rvalue::Aggregate(AggregateKind::Tuple, vec![]);
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
                        target: target
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
                            func: func,
                            args: args,
                            cleanup: None,
                            destination: Some((Lvalue::Local(new_temp), new_target))
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
        let span = self.promoted.span;
        let new_operand = Operand::Constant(Constant {
            span: span,
            ty: self.promoted.return_ty,
            literal: Literal::Promoted {
                index: Promoted::new(self.source.promoted.len())
            }
        });
        let mut rvalue = match candidate {
            Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                let ref mut statement = self.source[bb].statements[stmt_idx];
                match statement.kind {
                    StatementKind::Assign(_, ref mut rvalue) => {
                        mem::replace(rvalue, Rvalue::Use(new_operand))
                    }
                    _ => bug!()
                }
            }
            Candidate::ShuffleIndices(bb) => {
                match self.source[bb].terminator_mut().kind {
                    TerminatorKind::Call { ref mut args, .. } => {
                        Rvalue::Use(mem::replace(&mut args[2], new_operand))
                    }
                    _ => bug!()
                }
            }
        };
        self.visit_rvalue(&mut rvalue, Location {
            block: BasicBlock::new(0),
            statement_index: usize::MAX
        });

        self.assign(RETURN_POINTER, rvalue, span);
        self.source.promoted.push(self.promoted);
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn visit_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        if let Lvalue::Local(ref mut temp) = *lvalue {
            if self.source.local_kind(*temp) == LocalKind::Temp {
                *temp = self.promote_temp(*temp);
            }
        }
        self.super_lvalue(lvalue, context, location);
    }
}

pub fn promote_candidates<'a, 'tcx>(mir: &mut Mir<'tcx>,
                                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    mut temps: IndexVec<Local, TempState>,
                                    candidates: Vec<Candidate>) {
    // Visit candidates in reverse, in case they're nested.
    debug!("promote_candidates({:?})", candidates);
    for candidate in candidates.into_iter().rev() {
        let (span, ty) = match candidate {
            Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                let statement = &mir[bb].statements[stmt_idx];
                let dest = match statement.kind {
                    StatementKind::Assign(ref dest, _) => dest,
                    _ => {
                        span_bug!(statement.source_info.span,
                                  "expected assignment to promote");
                    }
                };
                if let Lvalue::Local(index) = *dest {
                    if temps[index] == TempState::PromotedOut {
                        // Already promoted.
                        continue;
                    }
                }
                (statement.source_info.span, dest.ty(mir, tcx).to_ty(tcx))
            }
            Candidate::ShuffleIndices(bb) => {
                let terminator = mir[bb].terminator();
                let ty = match terminator.kind {
                    TerminatorKind::Call { ref args, .. } => {
                        args[2].ty(mir, tcx)
                    }
                    _ => {
                        span_bug!(terminator.source_info.span,
                                  "expected simd_shuffleN call to promote");
                    }
                };
                (terminator.source_info.span, ty)
            }
        };

        // Declare return pointer local
        let initial_locals = iter::once(LocalDecl::new_return_pointer(ty)).collect();

        let mut promoter = Promoter {
            promoted: Mir::new(
                IndexVec::new(),
                Some(VisibilityScopeData {
                    span: span,
                    parent_scope: None
                }).into_iter().collect(),
                IndexVec::new(),
                ty,
                initial_locals,
                0,
                vec![],
                span
            ),
            source: mir,
            temps: &mut temps,
            keep_original: false
        };
        assert_eq!(promoter.new_block(), START_BLOCK);
        promoter.promote_candidate(candidate);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in mir.basic_blocks_mut() {
        block.statements.retain(|statement| {
            match statement.kind {
                StatementKind::Assign(Lvalue::Local(index), _) |
                StatementKind::StorageLive(Lvalue::Local(index)) |
                StatementKind::StorageDead(Lvalue::Local(index)) => {
                    !promoted(index)
                }
                _ => true
            }
        });
        let terminator = block.terminator_mut();
        match terminator.kind {
            TerminatorKind::Drop { location: Lvalue::Local(index), target, .. } => {
                if promoted(index) {
                    terminator.kind = TerminatorKind::Goto {
                        target: target
                    };
                }
            }
            _ => {}
        }
    }
}
