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
use rustc::mir::visit::{PlaceContext, MutatingUseContext, MutVisitor, Visitor};
use rustc::mir::traversal::ReversePostorder;
use rustc::ty::TyCtxt;
use syntax_pos::Span;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use std::{iter, mem, usize};

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
        debug!("is_promotable: self={:?}", self);
        if let TempState::Defined { .. } = *self {
            true
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
    body: &'tcx Body<'tcx>,
}

impl<'tcx> Visitor<'tcx> for TempCollector<'tcx> {
    fn visit_local(&mut self,
                   &index: &Local,
                   context: PlaceContext,
                   location: Location) {
        debug!("visit_local: index={:?} context={:?} location={:?}", index, context, location);
        // We're only interested in temporaries and the return place
        match self.body.local_kind(index) {
            | LocalKind::Temp
            | LocalKind::ReturnPointer
            => {},
            | LocalKind::Arg
            | LocalKind::Var
            => return,
        }

        // Ignore drops, if the temp gets promoted,
        // then it's constant and thus drop is noop.
        // Non-uses are also irrelevent.
        if context.is_drop() || !context.is_use() {
            debug!(
                "visit_local: context.is_drop={:?} context.is_use={:?}",
                context.is_drop(), context.is_use(),
            );
            return;
        }

        let temp = &mut self.temps[index];
        debug!("visit_local: temp={:?}", temp);
        if *temp == TempState::Undefined {
            match context {
                PlaceContext::MutatingUse(MutatingUseContext::Store) |
                PlaceContext::MutatingUse(MutatingUseContext::Call) => {
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
            // to promote mutable borrows of some ZSTs e.g., `&mut []`.
            let allowed_use = context.is_borrow() || context.is_nonmutating_use();
            debug!("visit_local: allowed_use={:?}", allowed_use);
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

pub fn collect_temps(body: &Body<'_>,
                     rpo: &mut ReversePostorder<'_, '_>) -> IndexVec<Local, TempState> {
    let mut collector = TempCollector {
        temps: IndexVec::from_elem(TempState::Undefined, &body.local_decls),
        span: body.span,
        body,
    };
    for (bb, data) in rpo {
        collector.visit_basic_block_data(bb, data);
    }
    collector.temps
}

struct Promoter<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    source: &'a mut Body<'tcx>,
    promoted: Body<'tcx>,
    temps: &'a mut IndexVec<Local, TempState>,

    /// If true, all nested temps are also kept in the
    /// source MIR, not moved to the promoted MIR.
    keep_original: bool,
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
            kind: StatementKind::Assign(Place::from(dest), box rvalue)
        });
    }

    /// Copies the initialization of this temp to the
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
            let (rvalue, source_info) = {
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
                    let unit = box Rvalue::Aggregate(box AggregateKind::Tuple, vec![]);
                    mem::replace(rhs, unit)
                }, statement.source_info)
            };

            let mut rvalue = *rvalue;
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
                TerminatorKind::Call { mut func, mut args, from_hir_call, .. } => {
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
                            destination: Some(
                                (Place::from(new_temp), new_target)
                            ),
                            from_hir_call,
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
        let mut operand = {
            let promoted = &mut self.promoted;
            let promoted_id = Promoted::new(self.source.promoted.len());
            let mut promoted_place = |ty, span| {
                promoted.span = span;
                promoted.local_decls[RETURN_PLACE] = LocalDecl::new_return_place(ty, span);
                Place::Base(
                    PlaceBase::Static(box Static{ kind: StaticKind::Promoted(promoted_id), ty })
                )
            };
            let (blocks, local_decls) = self.source.basic_blocks_and_local_decls_mut();
            match candidate {
                Candidate::Ref(loc) => {
                    let ref mut statement = blocks[loc.block].statements[loc.statement_index];
                    match statement.kind {
                        StatementKind::Assign(_, box Rvalue::Ref(_, _, ref mut place)) => {
                            // Find the underlying local for this (necessarily interior) borrow.
                            let mut place = place;
                            while let Place::Projection(ref mut proj) = *place {
                                assert_ne!(proj.elem, ProjectionElem::Deref);
                                place = &mut proj.base;
                            };

                            let ty = place.ty(local_decls, self.tcx).ty;
                            let span = statement.source_info.span;

                            Operand::Move(mem::replace(place, promoted_place(ty, span)))
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
                            let operand = Operand::Copy(promoted_place(ty, span));
                            mem::replace(&mut args[index], operand)
                        }
                        // We expected a `TerminatorKind::Call` for which we'd like to promote an
                        // argument. `qualify_consts` saw a `TerminatorKind::Call` here, but
                        // we are seeing a `Goto`. That means that the `promote_temps` method
                        // already promoted this call away entirely. This case occurs when calling
                        // a function requiring a constant argument and as that constant value
                        // providing a value whose computation contains another call to a function
                        // requiring a constant argument.
                        TerminatorKind::Goto { .. } => return,
                        _ => bug!()
                    }
                }
            }
        };

        assert_eq!(self.new_block(), START_BLOCK);
        self.visit_operand(&mut operand, Location {
            block: BasicBlock::new(0),
            statement_index: usize::MAX
        });

        let span = self.promoted.span;
        self.assign(RETURN_PLACE, Rvalue::Use(operand), span);
        self.source.promoted.push(self.promoted);
    }
}

/// Replaces all temporaries with their promoted counterparts.
impl<'a, 'tcx> MutVisitor<'tcx> for Promoter<'a, 'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        if self.source.local_kind(*local) == LocalKind::Temp {
            *local = self.promote_temp(*local);
        }
    }
}

pub fn promote_candidates<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    mut temps: IndexVec<Local, TempState>,
    candidates: Vec<Candidate>,
) {
    // Visit candidates in reverse, in case they're nested.
    debug!("promote_candidates({:?})", candidates);

    for candidate in candidates.into_iter().rev() {
        match candidate {
            Candidate::Ref(Location { block, statement_index }) => {
                match body[block].statements[statement_index].kind {
                    StatementKind::Assign(Place::Base(PlaceBase::Local(local)), _) => {
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


        // Declare return place local so that `mir::Body::new` doesn't complain.
        let initial_locals = iter::once(
            LocalDecl::new_return_place(tcx.types.never, body.span)
        ).collect();

        let promoter = Promoter {
            promoted: Body::new(
                IndexVec::new(),
                // FIXME: maybe try to filter this to avoid blowing up
                // memory usage?
                body.source_scopes.clone(),
                body.source_scope_local_data.clone(),
                IndexVec::new(),
                None,
                initial_locals,
                IndexVec::new(),
                0,
                vec![],
                body.span,
                vec![],
            ),
            tcx,
            source: body,
            temps: &mut temps,
            keep_original: false
        };
        promoter.promote_candidate(candidate);
    }

    // Eliminate assignments to, and drops of promoted temps.
    let promoted = |index: Local| temps[index] == TempState::PromotedOut;
    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| {
            match statement.kind {
                StatementKind::Assign(Place::Base(PlaceBase::Local(index)), _) |
                StatementKind::StorageLive(index) |
                StatementKind::StorageDead(index) => {
                    !promoted(index)
                }
                _ => true
            }
        });
        let terminator = block.terminator_mut();
        match terminator.kind {
            TerminatorKind::Drop { location: Place::Base(PlaceBase::Local(index)), target, .. } => {
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
