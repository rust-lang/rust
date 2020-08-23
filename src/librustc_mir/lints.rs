use rustc_data_structures::{fx::FxHashMap, stable_set::FxHashSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::FnKind;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::mir::{BasicBlock, Body, TerminatorKind, START_BLOCK};
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, AssocItem, AssocItemContainer, Instance, TyCtxt};
use rustc_session::lint::builtin::UNCONDITIONAL_RECURSION;
use rustc_span::{MultiSpan, Span};
use std::collections::VecDeque;
use std::{iter, mem, rc::Rc};
use ty::{subst::SubstsRef, WithOptConstParam};

const MAX_DEPTH: usize = 4;

crate fn check<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    if let Some(fn_like_node) = FnLikeNode::from_node(tcx.hir().get(hir_id)) {
        if let FnKind::Closure(_) = fn_like_node.kind() {
            // closures can't recur, so they don't matter.
            return;
        }

        debug!("MIR-linting {}", tcx.def_path_str(def_id.to_def_id()));

        // If this is trait/impl method, extract the trait's substs.
        let trait_substs = match tcx.opt_associated_item(def_id.to_def_id()) {
            Some(AssocItem {
                container: AssocItemContainer::TraitContainer(trait_def_id), ..
            }) => {
                let trait_substs_count = tcx.generics_of(*trait_def_id).count();
                &InternalSubsts::identity_for_item(tcx, def_id.to_def_id())[..trait_substs_count]
            }
            _ => &[],
        };

        // Keep track of instances we've already visited to avoid running in circles.
        let mut seen = FxHashSet::default();

        // Worklist of instances to search for recursive calls. Starts out with just the original
        // item.
        let mut worklist = VecDeque::new();
        worklist.push_back(Rc::new(Item {
            depth: 0,
            def_id: def_id.to_def_id(),
            // Initially, we use the identity substs, but calls inside the function can provide
            // concrete types for callees.
            substs: InternalSubsts::identity_for_item(tcx, def_id.to_def_id()),
            caller: None,
            caller_spans: &[],
        }));

        while let Some(item) = worklist.pop_front() {
            if !seen.insert((item.def_id, item.substs)) {
                // Already processed this instance.
                continue;
            }

            if item.depth > MAX_DEPTH {
                // Call stack too deep, bail out.
                continue;
            }

            // FIXME: Apply substs as appropriate to look into generic functions and trait methods.
            // This is not done at the moment, since `Instance::resolve` would be called with
            // partially monomorphized arguments, which would allow *resolving* the instance, but
            // not *normalizing* the result, causing ICEs.

            let param_env = tcx.param_env(item.def_id);

            debug!("processing callees of {} - {:?}", tcx.def_path_str(item.def_id), item);
            for &(callee_def_id, callee_substs, spans) in tcx.inevitable_calls(item.def_id) {
                let (call_fn_id, call_substs) =
                    match Instance::resolve(tcx, param_env, callee_def_id, callee_substs) {
                        Ok(Some(instance)) => {
                            // The precise callee is statically known. We only handle callees for
                            // which there's MIR available (those are the only ones that can cause
                            // cycles anyways).
                            if tcx.is_mir_available(instance.def_id()) {
                                debug!(
                                    "call target of {:?} {:?} is {:?}",
                                    callee_def_id, callee_substs, instance
                                );
                                (instance.def_id(), instance.substs)
                            } else {
                                debug!("no MIR for instance {:?}, skipping", instance);
                                continue;
                            }
                        }
                        _ if callee_def_id == def_id.to_def_id() => {
                            // If the instance fails to resolve, we might be a specializable or
                            // defaulted function. However, if we know we're calling *ourselves*
                            // only, we still lint, since any use without overriding the impl will
                            // result in self-recursion.
                            (callee_def_id, callee_substs)
                        }
                        _ => {
                            // Otherwise, this call does not have a statically known target, so we
                            // cannot lint it.
                            debug!(
                                "call target unknown: {:?} ({:?}) in {:#?}",
                                callee_def_id, callee_substs, param_env
                            );
                            continue;
                        }
                    };

                // We have found a self-call that we want to lint when a (resolved) `Callee` has our
                // `DefId` and (if it's a trait/impl method) the same trait-level substs.
                let is_self_call = call_fn_id == def_id.to_def_id()
                    && &call_substs[..trait_substs.len()] == trait_substs;

                if is_self_call {
                    item.lint_self_call(tcx, spans);
                    return;
                }

                // Put this callee at the end of the worklist.
                worklist.push_back(Rc::new(Item {
                    depth: item.depth + 1,
                    def_id: call_fn_id,
                    substs: call_substs,
                    caller: Some(item.clone()),
                    caller_spans: spans,
                }));
            }
        }
    }
}

/// A node in the call tree.
///
/// As the lint runs, we'll create a call tree as we look for cycles. We add items in a
/// breadth-first manner, so this does *not* create a stack.
#[derive(Clone, Debug)]
struct Item<'tcx> {
    /// Depth in the call stack/tree. Starts out at 0 for the root node, and is incremented for each
    /// level in the tree. This is used to limit the amount of work we do.
    depth: usize,
    /// The function being called.
    def_id: DefId,
    /// The substs applied to the called function.
    substs: SubstsRef<'tcx>,
    /// The `Item` from which the call is made. `None` if this is the tree root (in which case it
    /// describes the original function we're linting).
    caller: Option<Rc<Item<'tcx>>>,
    /// List of spans in `caller` that are the call-sites at which this particular `Item` is called.
    /// Empty if this is the root.
    caller_spans: &'tcx [Span],
}

impl<'tcx> Item<'tcx> {
    fn lint_self_call(&self, tcx: TyCtxt<'tcx>, final_spans: &[Span]) {
        // Collect the call stack, reversing the tree order.
        let mut cur = Rc::new(self.clone());
        let mut stack = vec![cur.clone()];
        while let Some(caller) = &cur.caller {
            stack.push(caller.clone());
            cur = caller.clone();
        }
        stack.reverse();

        let spans = stack
            .iter()
            .skip(1)
            .map(|item| item.caller_spans)
            .chain(iter::once(final_spans))
            .collect::<Vec<_>>();

        // First item is the original function we're linting.
        let target_def_id = stack[0].def_id;
        let hir_id = tcx.hir().local_def_id_to_hir_id(target_def_id.expect_local());
        let sp = tcx.sess.source_map().guess_head_span(tcx.hir().span_with_body(hir_id));
        tcx.struct_span_lint_hir(UNCONDITIONAL_RECURSION, hir_id, sp, |lint| {
            let mut db = lint.build("function cannot return without recursing");

            for (i, item) in stack.iter().enumerate() {
                let is_first = i == 0;
                let is_last = i == stack.len() - 1;

                let fn_span = item.def_id.as_local().map(|local| {
                    let hir_id = tcx.hir().local_def_id_to_hir_id(local);
                    let sp = tcx.sess.source_map().guess_head_span(tcx.hir().span(hir_id));
                    sp
                });

                // For each function, collect the contained call sites leading to the next function.
                let call_spans = match fn_span {
                    Some(fn_span) => {
                        let mut call_spans = MultiSpan::from_span(fn_span);
                        for &span in spans[i] {
                            let msg = if stack.len() == 1 {
                                "recursive call site"
                            } else if is_last {
                                "call completing the cycle"
                            } else {
                                "call into the next function in the cycle"
                            };
                            call_spans.push_span_label(span, msg.into());
                        }
                        call_spans
                    }
                    None => MultiSpan::new(),
                };

                // Put the collected labels on the corresponding function.
                if is_first {
                    db.span_label(fn_span.unwrap(), "cannot return without recursing");
                    for lab in call_spans.span_labels() {
                        if let Some(label) = lab.label {
                            db.span_label(lab.span, label);
                        }
                    }
                } else if fn_span.is_some() {
                    // We have useful spans in `call_spans`.
                    db.span_note(call_spans, "next function in the cycle");
                } else {
                    db.note(&format!(
                        "next function in the cycle is `{}`",
                        tcx.def_path_str(item.def_id)
                    ));
                }
            }

            if stack.len() == 1 {
                // A single self-calling function may be rewritten via `loop`.
                db.help("a `loop` may express intention better if this is on purpose");
            }

            db.emit();
        });
    }
}

/// Query provider.
///
/// This query is forced before `mir_validated` steals the `mir_built` results, so `mir_built` is
/// always available for local items here.
crate fn inevitable_calls<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: DefId,
) -> &'tcx [(DefId, SubstsRef<'tcx>, &'tcx [Span])] {
    tcx.arena.alloc_from_iter(
        find_inevitable_calls(tcx, key)
            .into_iter()
            .flatten()
            .map(|(callee, spans)| (callee.def_id, callee.substs, &*tcx.arena.alloc_slice(&spans))),
    )
}

/// Computes the set of callees that are known to be called whenever this function is entered.
fn find_inevitable_calls<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> Option<impl Iterator<Item = (Callee<'tcx>, Vec<Span>)>> {
    debug!("find_inevitable_calls({})", tcx.def_path_str(def_id));

    assert!(tcx.is_mir_available(def_id), "MIR unavailable for {:?}", def_id);

    let steal;
    let steal_ref;
    let body = match def_id.as_local() {
        Some(local) => {
            // Call the actual MIR building query for items in the local crate.

            // For some reason `is_mir_available` will return true for items where `mir_built`
            // will then panic with "can't build MIR for X" (for example, tuple struct/variant
            // ctors). Check that the item in question is an actual function-like thingy.
            let hir_id = tcx.hir().local_def_id_to_hir_id(local);
            match FnLikeNode::from_node(tcx.hir().get(hir_id)) {
                Some(_) => {
                    debug!("find_inevitable_calls: local node, using `mir_built`");
                    steal = tcx.mir_built(WithOptConstParam::unknown(local));
                    steal_ref = steal.borrow();
                    &*steal_ref
                }
                None => {
                    debug!("{:?} not an `FnLikeNode`, not linting", local);
                    return None;
                }
            }
        }
        None => {
            // Right now, external functions cannot cause cycles since we can't look into trait
            // impls or anything generic. Also, it turns out that `is_mir_available` is not
            // sufficient to determine whether `optimized_mir` will succeed (we hit "unwrap on a
            // None value" in the metadata decoder).
            debug!("external function {:?}, not linting", def_id);
            return None;
        }
    };

    Some(find_inevitable_calls_in_body(tcx, body))
}

fn find_inevitable_calls_in_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> impl Iterator<Item = (Callee<'tcx>, Vec<Span>)> {
    // Maps from BasicBlocks to the set of callees they are guaranteed to reach. Starts out as a map
    // from BasicBlocks whose terminators *are* calls.
    let mut inevitable_calls =
        IndexVec::from_elem_n(FxHashSet::default(), body.basic_blocks().len());
    let mut span_map: FxHashMap<Callee<'_>, Vec<_>> = FxHashMap::default();

    for (bb, callee, span) in collect_outgoing_calls(tcx, body) {
        inevitable_calls[bb].insert(callee);
        span_map.entry(callee).or_default().push(span);
    }

    let predecessors = body.predecessors();

    // Worklist of block to propagate inevitable callees into. Propagation runs backwards starting
    // at the call sites.
    let mut worklist: VecDeque<_> =
        inevitable_calls.indices().flat_map(|bb| &predecessors[bb]).collect();

    let mut successors = Vec::with_capacity(2);

    while let Some(&bb) = worklist.pop_front() {
        // Determine all "relevant" successors. We ignore successors only reached via unwinding.
        let terminator = body[bb].terminator();
        let relevant_successors = match &terminator.kind {
            TerminatorKind::Call { destination: None, .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop => None.into_iter().chain(&[]),
            TerminatorKind::SwitchInt { targets, .. } => None.into_iter().chain(targets),
            TerminatorKind::Goto { target }
            | TerminatorKind::Drop { target, .. }
            | TerminatorKind::DropAndReplace { target, .. }
            | TerminatorKind::Assert { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. }
            | TerminatorKind::Call { destination: Some((_, target)), .. } => {
                Some(target).into_iter().chain(&[])
            }
            TerminatorKind::InlineAsm { destination: Some(dest), .. } => {
                Some(dest).into_iter().chain(&[])
            }
            TerminatorKind::InlineAsm { destination: None, .. } => None.into_iter().chain(&[]),
            TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable => {
                // We propagate backwards, so these should never be encountered here.
                unreachable!("unexpected terminator {:?}", terminator.kind)
            }
        };

        successors.clear();
        successors.extend(relevant_successors.copied());

        let mut dest_set = mem::take(&mut inevitable_calls[bb]);

        let changed = propagate_successors(&mut dest_set, &inevitable_calls, &successors);

        inevitable_calls[bb] = dest_set;
        if changed {
            // `bb`s inevitable callees were modified, so propagate that backwards.
            worklist.extend(&predecessors[bb]);
        }
    }

    mem::take(&mut inevitable_calls[START_BLOCK]).into_iter().map(move |callee| {
        let spans = &span_map[&callee];
        (callee, spans.clone())
    })
}

/// Propagates inevitable calls from `successors` into their predecessor's `dest_set`.
///
/// Returns `true` if `dest_set` was changed.
fn propagate_successors<'tcx>(
    dest_set: &mut FxHashSet<Callee<'tcx>>,
    inevitable_calls: &IndexVec<BasicBlock, FxHashSet<Callee<'tcx>>>,
    successors: &[BasicBlock],
) -> bool {
    let len = dest_set.len();

    match successors {
        [successor] => {
            // If there's only one successor, just add all of its calls to `dest_set`.
            dest_set.extend(inevitable_calls[*successor].iter().copied());
        }
        _ => {
            // All callees that are guaranteed to be reached by every successor will also be reached by
            // `bb`. Compute the intersection.
            // For efficiency, we initially only consider the smallest set.
            let (smallest_successor_set_index, smallest_successor_set_bb) = match successors
                .iter()
                .enumerate()
                .min_by_key(|(_, &bb)| inevitable_calls[bb].len())
            {
                Some((i, bb)) => (i, *bb),
                None => return false, // No callees will be added
            };

            for callee in &inevitable_calls[smallest_successor_set_bb] {
                if dest_set.contains(callee) {
                    continue;
                }

                // `callee` must be contained in *every* successor's set.
                let add = successors.iter().enumerate().all(|(i, bb)| {
                    if i == smallest_successor_set_index {
                        return true;
                    }
                    let callee_set = &inevitable_calls[*bb];
                    callee_set.contains(callee)
                });

                if add {
                    dest_set.insert(callee.clone());
                }
            }
        }
    }

    dest_set.len() != len
}

/// Information about a callee tracked by this algorithm.
#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
struct Callee<'tcx> {
    def_id: DefId,
    substs: SubstsRef<'tcx>,
}

/// Walks `body` to collect all known callees.
fn collect_outgoing_calls<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = (BasicBlock, Callee<'tcx>, Span)> + 'a {
    body.basic_blocks().iter_enumerated().filter_map(move |(bb, data)| {
        if let TerminatorKind::Call { func, .. } = &data.terminator().kind {
            let func_ty = func.ty(body, tcx);
            if let ty::FnDef(def_id, substs) = func_ty.kind {
                let callee = Callee { def_id, substs };
                let span = data.terminator().source_info.span;
                return Some((bb, callee, span));
            }
        }

        None
    })
}
