use super::MirPass;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    AggregateKind, BasicBlock, Body, Local, Location, Operand, Place, Rvalue, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::MaybeBorrowedLocals;
use rustc_mir_dataflow::Analysis;
use rustc_session::Session;

use super::simplify;
use super::ssa::SsaLocals;

/// # Overview
///
/// This pass looks to optimize a pattern in MIR where variants of an aggregate
/// are constructed in one or more blocks with the same successor and then that
/// aggregate/discriminant is switched on in that successor block, in which case
/// we can remove the switch on the discriminant because we statically know
/// what target block will be taken for each variant.
///
/// Note that an aggregate which is returned from a function call or passed as
/// an argument is not viable for this optimization because we do not statically
/// know the discriminant/variant of the aggregate.
///
/// For example, the following CFG:
/// ```text
///     x = Foo::A(y); ---       Foo::A ---> ...
///    /                  \            /
/// ...                    --> switch x
///    \                  /            \
///     x = Foo::B(y); ---       Foo::B ---> ...
/// ```
/// would become:
/// ```text
///     x = Foo::A(y); --------- Foo::A ---> ...
///    /
/// ...
///    \
///     x = Foo::B(y); --------- Foo::B ---> ...
/// ```
///
/// # Soundness
///
///  - If the discriminant being switched on is not SSA, or if the aggregate is
///    mutated before the discriminant is assigned, the optimization cannot be
///    applied because we no longer statically know what variant the aggregate
///    could be, or what discriminant is being switched on.
///
///  - If the discriminant is borrowed before being switched on, or the aggregate
///    is borrowed before the discriminant is assigned, we also cannot optimize due
///    to the possibilty stated in the first paragraph.
///
///  - An aggregate being constructed has a known variant, and if it is not borrowed
///    or mutated before being switched on, then it does not actually need a runtime
///    switch on the discriminant (aka variant) of said aggregate.
///
pub struct SimplifyStaticSwitch;

impl<'tcx> MirPass<'tcx> for SimplifyStaticSwitch {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("Running SimplifyStaticSwitch on {:?}", body.source.def_id());

        let ssa_locals = SsaLocals::new(body);
        if simplify_static_switches(tcx, body, &ssa_locals) {
            simplify::remove_dead_blocks(tcx, body);
        }
    }
}

#[instrument(level = "debug", skip_all, ret)]
fn simplify_static_switches<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    ssa_locals: &SsaLocals,
) -> bool {
    let dominators = body.basic_blocks.dominators();
    let predecessors = body.basic_blocks.predecessors();
    let mut discriminants = FxHashMap::default();
    let mut static_switches = FxHashMap::default();
    let mut borrowed_locals =
        MaybeBorrowedLocals.into_engine(tcx, body).iterate_to_fixpoint().into_results_cursor(body);
    for (switched, rvalue, location) in ssa_locals.assignments(body) {
        let Rvalue::Discriminant(discr) = rvalue else {
            continue
        };

        borrowed_locals.seek_after_primary_effect(location);
        // If `discr` was borrowed before its discriminant was assigned to `switched`,
        // or if it was borrowed in the assignment, we cannot optimize.
        if borrowed_locals.contains(discr.local) {
            debug!("The aggregate: {discr:?} was borrowed before its discriminant was read");
            continue;
        }

        let Location { block, statement_index } = location;
        let mut finder = MutatedLocalFinder { local: discr.local, mutated: false };
        for (statement_index, statement) in body.basic_blocks[block]
            .statements
            .iter()
            .enumerate()
            .take_while(|&(index, _)| index != statement_index)
        {
            finder.visit_statement(statement, Location { block, statement_index });
        }

        if finder.mutated {
            debug!("The aggregate: {discr:?} was mutated before its discriminant was read");
            continue;
        }

        // If `switched` is borrowed by the time we actually switch on it, we also cannot optimize.
        borrowed_locals.seek_to_block_end(block);
        if borrowed_locals.contains(switched) {
            debug!("The local: {switched:?} was borrowed before being switched on");
            continue;
        }

        discriminants.insert(
            switched,
            Discriminant {
                block,
                discr: discr.local,
                exclude: if ssa_locals.num_direct_uses(switched) == 1 {
                    // If there is only one direct use of `switched` we do not need to keep
                    // it around because the only use is in the switch.
                    Some(statement_index)
                } else {
                    None
                },
            },
        );
    }

    if discriminants.is_empty() {
        debug!("No SSA locals were assigned a discriminant");
        return false;
    }

    for (switched, Discriminant { discr, block, exclude }) in discriminants {
        let data = &body.basic_blocks[block];
        if data.is_cleanup {
            continue;
        }

        let predecessors = &predecessors[block];
        if predecessors.is_empty() {
            continue;
        }

        if predecessors.iter().any(|&pred| {
            // If we find a backedge from: `pred -> block`, this indicates that
            // `block` is a loop header. To avoid creating irreducible CFGs we do
            // not thread through loop headers.
            dominators.dominates(block, pred)
        }) {
            debug!("Unable to thread through loop header: {block:?}");
            continue;
        }

        let terminator = data.terminator();
        let TerminatorKind::SwitchInt {
            discr: Operand::Copy(place) | Operand::Move(place),
            targets
        } = &terminator.kind else {
            continue
        };

        if place.local != switched {
            continue;
        }

        let mut finder = MutatedLocalFinder { local: discr, mutated: false };
        'preds: for &pred in predecessors {
            let data = &body.basic_blocks[pred];
            let terminator = data.terminator();
            let TerminatorKind::Goto { .. } = terminator.kind else {
                continue
            };

            for (statement_index, statement) in data.statements.iter().enumerate().rev() {
                match statement.kind {
                    StatementKind::SetDiscriminant { box place, variant_index: variant }
                    | StatementKind::Assign(box (
                        place,
                        Rvalue::Aggregate(box AggregateKind::Adt(_, variant, ..), ..),
                    )) if place.local == discr => {
                        if finder.mutated {
                            debug!(
                                "The discriminant: {discr:?} was mutated in predecessor: {pred:?}"
                            );
                            // We can't optimize this predecessor, so try the next one.
                            finder.mutated = false;

                            continue 'preds;
                        }

                        let discr_ty = body.local_decls[discr].ty;
                        if let Some(discr) = discr_ty.discriminant_for_variant(tcx, variant) {
                            debug!(
                                "{pred:?}: {place:?} = {discr_ty:?}::{variant:?}; goto -> {block:?}",
                            );

                            let target = targets.target_for_value(discr.val);
                            static_switches
                                .entry(block)
                                .and_modify(|static_switches: &mut &mut [StaticSwitch]| {
                                    if static_switches.iter_mut().all(|switch| {
                                        if switch.pred == pred {
                                            switch.target = target;
                                            false
                                        } else {
                                            true
                                        }
                                    }) {
                                        *static_switches =
                                            tcx.arena.alloc_from_iter(
                                                static_switches.iter().copied().chain([
                                                    StaticSwitch { pred, target, exclude },
                                                ]),
                                            );
                                    }
                                })
                                .or_insert_with(|| {
                                    tcx.arena.alloc([StaticSwitch { pred, target, exclude }])
                                });
                        }

                        continue 'preds;
                    }
                    _ if finder.mutated => {
                        debug!("The discriminant: {discr:?} was mutated in predecessor: {pred:?}");
                        // Note that the discriminant could have been mutated in one predecessor
                        // but not the others, in which case only the predecessors which did not mutate
                        // the discriminant can be optimized.
                        finder.mutated = false;

                        continue 'preds;
                    }
                    _ => finder.visit_statement(statement, Location { block, statement_index }),
                }
            }
        }
    }

    if static_switches.is_empty() {
        debug!("No static switches were found in the current body");
        return false;
    }

    let basic_blocks = body.basic_blocks.as_mut();
    let num_switches: usize = static_switches.iter().map(|(_, switches)| switches.len()).sum();
    for (block, static_switches) in static_switches {
        for switch in static_switches {
            debug!("{block:?}: Removing static switch: {switch:?}");

            // We use the SSA, to destroy the SSA.
            let data = {
                let (block, pred) = basic_blocks.pick2_mut(block, switch.pred);
                match switch.exclude {
                    Some(exclude) => {
                        pred.statements.extend(block.statements.iter().enumerate().filter_map(
                            |(index, statement)| {
                                if index == exclude { None } else { Some(statement.clone()) }
                            },
                        ));
                    }
                    None => pred.statements.extend_from_slice(&block.statements),
                }
                pred
            };
            let terminator = data.terminator_mut();

            // Make sure that we have not overwritten the terminator and it is still
            // a `goto -> block`.
            assert_eq!(terminator.kind, TerminatorKind::Goto { target: block });
            // Something to be noted is that, this creates an edge from: `pred -> target`,
            // and because we ensure that we do not thread through any loop headers, meaning
            // it is not part of a loop, this edge will only ever appear once in the CFG.
            terminator.kind = TerminatorKind::Goto { target: switch.target };
        }
    }

    debug!("Removed {num_switches} static switches from: {:?}", body.source.def_id());
    true
}

#[derive(Debug, Copy, Clone)]
struct StaticSwitch {
    pred: BasicBlock,
    target: BasicBlock,
    exclude: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
struct Discriminant {
    discr: Local,
    block: BasicBlock,
    exclude: Option<usize>,
}

struct MutatedLocalFinder {
    local: Local,
    mutated: bool,
}

impl<'tcx> Visitor<'tcx> for MutatedLocalFinder {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
        if self.local == place.local && let PlaceContext::MutatingUse(..) = context {
            self.mutated = true;
        }
    }
}
