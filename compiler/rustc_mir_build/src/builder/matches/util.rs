use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::*;
use rustc_middle::ty::Ty;
use rustc_span::Span;
use tracing::debug;

use crate::builder::Builder;
use crate::builder::expr::as_place::PlaceBase;
use crate::builder::matches::{Binding, Candidate, FlatPat, MatchPairTree, TestCase};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Creates a false edge to `imaginary_target` and a real edge to
    /// real_target. If `imaginary_target` is none, or is the same as the real
    /// target, a Goto is generated instead to simplify the generated MIR.
    pub(crate) fn false_edges(
        &mut self,
        from_block: BasicBlock,
        real_target: BasicBlock,
        imaginary_target: BasicBlock,
        source_info: SourceInfo,
    ) {
        if imaginary_target != real_target {
            self.cfg.terminate(
                from_block,
                source_info,
                TerminatorKind::FalseEdge { real_target, imaginary_target },
            );
        } else {
            self.cfg.goto(from_block, source_info, real_target)
        }
    }
}

/// Determine the set of places that have to be stable across match guards.
///
/// Returns a list of places that need a fake borrow along with a local to store it.
///
/// Match exhaustiveness checking is not able to handle the case where the place being matched on is
/// mutated in the guards. We add "fake borrows" to the guards that prevent any mutation of the
/// place being matched. There are a some subtleties:
///
/// 1. Borrowing `*x` doesn't prevent assigning to `x`. If `x` is a shared reference, the borrow
///    isn't even tracked. As such we have to add fake borrows of any prefixes of a place.
/// 2. We don't want `match x { (Some(_), _) => (), .. }` to conflict with mutable borrows of `x.1`, so we
///    only add fake borrows for places which are bound or tested by the match.
/// 3. We don't want `match x { Some(_) => (), .. }` to conflict with mutable borrows of `(x as
///    Some).0`, so the borrows are a special shallow borrow that only affects the place and not its
///    projections.
///    ```rust
///    let mut x = (Some(0), true);
///    match x {
///        (Some(_), false) => {}
///        _ if { if let Some(ref mut y) = x.0 { *y += 1 }; true } => {}
///        _ => {}
///    }
///    ```
/// 4. The fake borrows may be of places in inactive variants, e.g. here we need to fake borrow `x`
///    and `(x as Some).0`, but when we reach the guard `x` may not be `Some`.
///    ```rust
///    let mut x = (Some(Some(0)), true);
///    match x {
///        (Some(Some(_)), false) => {}
///        _ if { if let Some(Some(ref mut y)) = x.0 { *y += 1 }; true } => {}
///        _ => {}
///    }
///    ```
///    So it would be UB to generate code for the fake borrows. They therefore have to be removed by
///    a MIR pass run after borrow checking.
pub(super) fn collect_fake_borrows<'tcx>(
    cx: &mut Builder<'_, 'tcx>,
    candidates: &[Candidate<'tcx>],
    temp_span: Span,
    scrutinee_base: PlaceBase,
) -> Vec<(Place<'tcx>, Local, FakeBorrowKind)> {
    if candidates.iter().all(|candidate| !candidate.has_guard) {
        // Fake borrows are only used when there is a guard.
        return Vec::new();
    }
    let mut collector =
        FakeBorrowCollector { cx, scrutinee_base, fake_borrows: FxIndexMap::default() };
    for candidate in candidates.iter() {
        collector.visit_candidate(candidate);
    }
    let fake_borrows = collector.fake_borrows;
    debug!("add_fake_borrows fake_borrows = {:?}", fake_borrows);
    let tcx = cx.tcx;
    fake_borrows
        .iter()
        .map(|(matched_place, borrow_kind)| {
            let fake_borrow_deref_ty = matched_place.ty(&cx.local_decls, tcx).ty;
            let fake_borrow_ty =
                Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, fake_borrow_deref_ty);
            let mut fake_borrow_temp = LocalDecl::new(fake_borrow_ty, temp_span);
            fake_borrow_temp.local_info = ClearCrossCrate::Set(Box::new(LocalInfo::FakeBorrow));
            let fake_borrow_temp = cx.local_decls.push(fake_borrow_temp);
            (*matched_place, fake_borrow_temp, *borrow_kind)
        })
        .collect()
}

pub(super) struct FakeBorrowCollector<'a, 'b, 'tcx> {
    cx: &'a mut Builder<'b, 'tcx>,
    /// Base of the scrutinee place. Used to distinguish bindings inside the scrutinee place from
    /// bindings inside deref patterns.
    scrutinee_base: PlaceBase,
    /// Store for each place the kind of borrow to take. In case of conflicts, we take the strongest
    /// borrow (i.e. Deep > Shallow).
    /// Invariant: for any place in `fake_borrows`, all the prefixes of this place that are
    /// dereferences are also borrowed with the same of stronger borrow kind.
    fake_borrows: FxIndexMap<Place<'tcx>, FakeBorrowKind>,
}

impl<'a, 'b, 'tcx> FakeBorrowCollector<'a, 'b, 'tcx> {
    // Fake borrow this place and its dereference prefixes.
    fn fake_borrow(&mut self, place: Place<'tcx>, kind: FakeBorrowKind) {
        if self.fake_borrows.get(&place).is_some_and(|k| *k >= kind) {
            return;
        }
        self.fake_borrows.insert(place, kind);
        // Also fake borrow the prefixes of any fake borrow.
        self.fake_borrow_deref_prefixes(place, kind);
    }

    // Fake borrow the prefixes of this place that are dereferences.
    fn fake_borrow_deref_prefixes(&mut self, place: Place<'tcx>, kind: FakeBorrowKind) {
        for (place_ref, elem) in place.as_ref().iter_projections().rev() {
            if let ProjectionElem::Deref = elem {
                // Insert a shallow borrow after a deref. For other projections the borrow of
                // `place_ref` will conflict with any mutation of `place.base`.
                let place = place_ref.to_place(self.cx.tcx);
                if self.fake_borrows.get(&place).is_some_and(|k| *k >= kind) {
                    return;
                }
                self.fake_borrows.insert(place, kind);
            }
        }
    }

    fn visit_candidate(&mut self, candidate: &Candidate<'tcx>) {
        for binding in &candidate.extra_data.bindings {
            if let super::SubpatternBindings::One(binding) = binding {
                self.visit_binding(binding);
            }
        }
        for match_pair in &candidate.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_flat_pat(&mut self, flat_pat: &FlatPat<'tcx>) {
        for binding in &flat_pat.extra_data.bindings {
            if let super::SubpatternBindings::One(binding) = binding {
                self.visit_binding(binding);
            }
        }
        for match_pair in &flat_pat.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_match_pair(&mut self, match_pair: &MatchPairTree<'tcx>) {
        if let TestCase::Or { pats, .. } = &match_pair.test_case {
            for flat_pat in pats.iter() {
                self.visit_flat_pat(flat_pat)
            }
        } else if matches!(match_pair.test_case, TestCase::Deref { .. }) {
            // The subpairs of a deref pattern are all places relative to the deref temporary, so we
            // don't fake borrow them. Problem is, if we only shallowly fake-borrowed
            // `match_pair.place`, this would allow:
            // ```
            // let mut b = Box::new(false);
            // match b {
            //     deref!(true) => {} // not reached because `*b == false`
            //     _ if { *b = true; false } => {} // not reached because the guard is `false`
            //     deref!(false) => {} // not reached because the guard changed it
            //     // UB because we reached the unreachable.
            // }
            // ```
            // Hence we fake borrow using a deep borrow.
            if let Some(place) = match_pair.place {
                self.fake_borrow(place, FakeBorrowKind::Deep);
            }
        } else {
            // Insert a Shallow borrow of any place that is switched on.
            if let Some(place) = match_pair.place {
                self.fake_borrow(place, FakeBorrowKind::Shallow);
            }

            for subpair in &match_pair.subpairs {
                self.visit_match_pair(subpair);
            }
        }
    }

    fn visit_binding(&mut self, Binding { source, .. }: &Binding<'tcx>) {
        if let PlaceBase::Local(l) = self.scrutinee_base
            && l != source.local
        {
            // The base of this place is a temporary created for deref patterns. We don't emit fake
            // borrows for these as they are not initialized in all branches.
            return;
        }

        // Insert a borrows of prefixes of places that are bound and are
        // behind a dereference projection.
        //
        // These borrows are taken to avoid situations like the following:
        //
        // match x[10] {
        //     _ if { x = &[0]; false } => (),
        //     y => (), // Out of bounds array access!
        // }
        //
        // match *x {
        //     // y is bound by reference in the guard and then by copy in the
        //     // arm, so y is 2 in the arm!
        //     y if { y == 1 && (x = &2) == () } => y,
        //     _ => 3,
        // }
        //
        // We don't just fake borrow the whole place because this is allowed:
        // match u {
        //     _ if { u = true; false } => (),
        //     x => (),
        // }
        self.fake_borrow_deref_prefixes(*source, FakeBorrowKind::Shallow);
    }
}

#[must_use]
pub(crate) fn ref_pat_borrow_kind(ref_mutability: Mutability) -> BorrowKind {
    match ref_mutability {
        Mutability::Mut => BorrowKind::Mut { kind: MutBorrowKind::Default },
        Mutability::Not => BorrowKind::Shared,
    }
}
