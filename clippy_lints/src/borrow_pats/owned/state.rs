#![warn(unused)]

use rustc_index::bit_set::GrowableBitSet;

use crate::borrow_pats::MyStateInfo;

use super::super::prelude::*;
use super::OwnedPat;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateInfo<'tcx> {
    bb: BasicBlock,
    // pub prev_state: (State, BasicBlock),
    state: SmallVec<[(State, BasicBlock); 4]>,
    /// This is a set of values that *might* contain the owned value.
    /// MIR has this *beautiful* habit of moving stuff into anonymous
    /// locals first before using it further.
    anons: FxHashMap<Local, AnonStorage<'tcx>>,
    containers: FxHashMap<Local, ContainerInfo>,
    /// This set contains borrows, these are often used for temporary
    /// borrows
    ///
    /// **Note 1**: Named borrows can be created in two ways (Because of course
    /// they can...)
    /// ```
    /// // From: `mut_named_ref_non_kill`
    /// //    let mut x = 1;
    /// //    let mut p: &u32 = &x;
    /// _4 = &_1
    /// _3 = &(*_4)
    ///
    /// // From: `call_extend_named`
    /// //    let data = String::new();
    /// //    let loan = &data;
    /// _2 = &_3
    /// ```
    ///
    /// **Note 2**: Correction there are three ways to created named borrows...
    /// Not sure why but let's take `mut_named_ref_non_kill` as and example for `y`
    ///
    /// ```
    /// // y     => _2
    /// // named => _3
    /// _8 = &_2
    /// _7 = &(*_8)
    /// _3 = move _7
    /// ```
    ///
    /// **Note 3**: If I can confirm that these borrows are always used for
    /// temporary borrows, it might be possible to prevent tracking them
    /// to safe some performance. (Confirmed, that they are not just
    /// used for temp borrows :D)
    borrows: FxHashMap<Local, BorrowInfo<'tcx>>,
    /// This tracks mut borrows, which might be used for two phased borrows.
    /// Based on the docs, it sounds like there can always only be one. Let's
    /// use an option and cry when it fails.
    ///
    /// See: <https://rustc-dev-guide.rust-lang.org/borrow_check/two_phase_borrows.html>
    phase_borrow: Vec<(Local, Place<'tcx>)>,
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct AnonStorage<'tcx> {
    places: SmallVec<[Place<'tcx>; 1]>,
}

impl<'tcx> AnonStorage<'tcx> {
    /// The first value indicates that this contains the whole palce,
    /// the second one that this contains a part. These two are not
    /// mutually exclusive
    pub fn place_props(&self) -> (bool, bool) {
        self.places.iter().fold((false, false), |(is_all, is_part), place| {
            (is_all || place.just_local(), is_part || place.is_part())
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ContainerInfo {
    content: FxHashSet<ContainerContent>,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ContainerContent {
    Loan,
    LoanMut,
    PartLoan,
    PartLoanMut,
    Owned,
    Part,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BorrowInfo<'tcx> {
    /// The place that is being borrowed
    pub broker: Place<'tcx>,
    /// This is the mutability of the original borrow. If we have a double borrow, like this:
    /// ```
    /// let mut data = String::new();
    ///
    /// //                Loan 1
    /// //                vvvvv
    /// let double_ref = &&mut data;
    /// //               ^
    /// //               Loan 2 (Mutable, since loan 1 is mut)
    /// ```
    pub muta: Mutability,
    pub kind: BroKind,
}

impl<'tcx> BorrowInfo<'tcx> {
    pub fn new(broker: Place<'tcx>, muta: Mutability, kind: BroKind) -> Self {
        Self { broker, muta, kind }
    }

    pub fn copy_with(&self, kind: BroKind) -> Self {
        Self::new(self.broker, self.muta, kind)
    }

    /// The first value indicates that this contains the whole palce,
    /// the second one that this contains a part. These two are not
    /// mutually exclusive
    pub fn borrowed_props(&self) -> (bool, bool) {
        if matches!(self.kind, BroKind::Dep) {
            (false, false)
        } else {
            (self.broker.just_local(), self.broker.is_part())
        }
    }

    pub fn as_content(&self) -> SmallVec<[ContainerContent; 1]> {
        let (is_all, is_part) = self.borrowed_props();
        let mut vec = SmallVec::new();
        if is_all {
            if matches!(self.muta, Mutability::Not) {
                vec.push(ContainerContent::Loan);
            }
            if matches!(self.muta, Mutability::Mut) {
                vec.push(ContainerContent::LoanMut);
            }
        }
        if is_part {
            if matches!(self.muta, Mutability::Not) {
                vec.push(ContainerContent::PartLoan);
            }
            if matches!(self.muta, Mutability::Mut) {
                vec.push(ContainerContent::PartLoanMut);
            }
        }
        vec
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Default)]
pub enum State {
    #[default]
    None,
    Empty,
    Filled,
    Moved,
    Dropped,
    MaybeFilled,
}

#[expect(unused)]
enum Event<'tcx> {
    Init,
    Loan,
    Mutated,
    // Moved or Dropped
    Moved(Place<'tcx>),
    Drop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum BroKind {
    Anon,
    Named,
    Dep,
}

impl<'tcx> StateInfo<'tcx> {
    pub fn prev_state(&self) -> Option<State> {
        if self.state.len() >= 2 {
            Some(self.state[self.state.len() - 2].0)
        } else {
            None
        }
    }

    pub fn state(&self) -> State {
        if let Some((state, _)) = self.state.last() {
            *state
        } else {
            unreachable!("State should always be filled: {self:#?}")
        }
    }

    pub fn set_state(&mut self, state: State) {
        self.state.push((state, self.bb));
    }

    /// Retruns true if this state contains valid data, which can be dropped or moved.
    pub fn validity(&self) -> Validity {
        match self.state() {
            State::None => unreachable!(),
            State::Filled => Validity::Valid,
            State::MaybeFilled => Validity::Maybe,
            State::Empty | State::Moved | State::Dropped => Validity::Not,
        }
    }
    /// Notifies the state that a local has been killed
    pub fn kill_local(&mut self, local: Local) {
        // self.anons.remove(&local);
        self.borrows.remove(&local);
        self.phase_borrow.retain(|(phase_local, _place)| *phase_local != local);
        self.containers.remove(&local);
    }

    pub fn add_anon(&mut self, anon: Local, src: Place<'tcx>) {
        self.anons.entry(anon).or_default().places.push(src);
    }

    pub fn add_anon_places(&mut self, anon: Local, places: AnonStorage<'tcx>) {
        let old_places = self.anons.insert(anon, places);
        assert!(old_places.is_none(), "Have fun debugging this one");
    }

    #[expect(unused)]
    pub fn add_container(&mut self, anon: Local, info: ContainerInfo) {
        self.containers
            .entry(anon)
            .and_modify(|other| other.content.extend(info.content.iter()))
            .or_insert(info);
    }

    /// This tries to remove the given place from the known anons that hold this value.
    /// It will retrun `true`, if the removal was successfull.
    /// Places with projections will be ignored.
    pub fn remove_anon(&mut self, anon: &Place<'_>) -> Option<AnonStorage<'tcx>> {
        let found = self.remove_anon_place(anon);
        // assert!(found.is_none() || anon.just_local(), "{self:#?} - {anon:#?}");
        found
    }

    pub fn remove_anon_place(&mut self, anon: &Place<'_>) -> Option<AnonStorage<'tcx>> {
        self.anons.remove(&anon.local)
    }

    /// This clears this state. The `state` field has to be set afterwards
    pub fn clear(&mut self, new_state: State) {
        self.anons.clear();
        self.borrows.clear();
        self.phase_borrow.clear();

        self.state.push((new_state, self.bb));
    }

    pub fn add_assign(&mut self, place: Place<'tcx>, pats: &mut BTreeSet<OwnedPat>) {
        let is_override = match self.state() {
            // No-op the most normal and simple state
            State::Moved | State::Empty => false,

            State::Dropped => {
                // A manual drop has `Moved` as the previous state
                if matches!(self.prev_state(), Some(State::Filled | State::MaybeFilled)) {
                    pats.insert(OwnedPat::DropForReplacement);
                    true
                } else {
                    false
                }
            },

            // Filled should only ever be the case for !Drop types
            State::Filled | State::MaybeFilled => true,

            State::None => unreachable!(),
        };
        if place.just_local() {
            if is_override {
                pats.insert(OwnedPat::Overwrite);
            }
            // Regardless of the original state, we clear everything else
            self.clear(State::Filled);
        } else if place.is_part() {
            if is_override {
                pats.insert(OwnedPat::OverwritePart);
            }
        } else {
            unreachable!();
        }
    }

    pub fn add_borrow(
        &mut self,
        borrow: Place<'tcx>,
        broker: Place<'tcx>,
        kind: BorrowKind,
        bro_kind: Option<BroKind>,
        info: &AnalysisInfo<'tcx>,
        pats: &mut BTreeSet<OwnedPat>,
    ) {
        self.update_bros(broker, kind.mutability(), info);

        if matches!(kind, BorrowKind::Shared)
            && self
                .phase_borrow
                .iter()
                .any(|(_loc, phase_place)| info.places_conflict(*phase_place, broker))
        {
            pats.insert(OwnedPat::TwoPhasedBorrow);
            info.stats.borrow_mut().owned.two_phased_borrows += 1;
        }

        let (is_all, is_part) = (broker.just_local(), broker.is_part());

        let is_named = matches!(info.locals[borrow.local].kind, LocalKind::UserVar(..));
        if is_named {
            if matches!(kind, BorrowKind::Shared) {
                info.stats.borrow_mut().owned.named_borrow_count += 1;
                if is_all {
                    pats.insert(OwnedPat::NamedBorrow);
                } else if is_part {
                    pats.insert(OwnedPat::PartNamedBorrow);
                } else {
                    unreachable!();
                }
            } else {
                info.stats.borrow_mut().owned.named_borrow_mut_count += 1;
                if is_all {
                    pats.insert(OwnedPat::NamedBorrowMut);
                } else if is_part {
                    pats.insert(OwnedPat::PartNamedBorrowMut);
                } else {
                    unreachable!();
                }
            }
        }

        let bro_kind = if let Some(bro_kind) = bro_kind {
            bro_kind
        } else if is_named {
            BroKind::Named
        } else {
            BroKind::Anon
        };

        // So: It turns out that MIR is an inconsisten hot mess. Two-Phase-Borrows are apparently
        // allowed to violate rust borrow semantics...
        //
        // Simple example: `x.push(x.len())`
        if is_named {
            // Mut loans can also be used for two-phased-borrows, but only with themselfs.
            // Taking the mut loan and the owned value failes.
            //
            // ```
            // fn test(mut vec: Vec<usize>) {
            //     let loan = &mut vec;
            //     loan.push(vec.len());
            // }
            // ```
            //
            // The two-phased-borrow will be detected by the owned reference. So we can
            // ignore it here :D
            self.borrows
                .insert(borrow.local, BorrowInfo::new(broker, kind.mutability(), bro_kind));
        } else {
            assert!(borrow.just_local());
            if kind.allows_two_phase_borrow() {
                self.phase_borrow.push((borrow.local, broker));
            } else {
                self.borrows
                    .insert(borrow.local, BorrowInfo::new(broker, kind.mutability(), bro_kind));
            }
        }
    }

    /// This function informs the state, that a local loan was just copied.
    pub fn add_ref_copy(
        &mut self,
        dst: Place<'tcx>,
        src: Place<'tcx>,
        info: &AnalysisInfo<'tcx>,
        pats: &mut BTreeSet<OwnedPat>,
    ) {
        self.add_ref_dep(dst, src, info, pats);
    }
    /// This function informs the state that a ref to a ref was created
    pub fn add_ref_ref(
        &mut self,
        dst: Place<'tcx>,
        src: Place<'tcx>,
        info: &AnalysisInfo<'tcx>,
        pats: &mut BTreeSet<OwnedPat>,
    ) {
        self.add_ref_dep(dst, src, info, pats);
    }
    /// If `kind` is empty it indicates that the mutability of `src` should be taken
    fn add_ref_dep(
        &mut self,
        dst: Place<'tcx>,
        src: Place<'tcx>,
        info: &AnalysisInfo<'tcx>,
        pats: &mut BTreeSet<OwnedPat>,
    ) {
        // This function has to share quite some magic with `add_borrow` but
        // again is different enough that they can't be merged directly AFAIK

        let Some(bro_info) = self.borrows.get(&src.local).copied() else {
            return;
        };

        // It looks like loans preserve the mutability of th copy. This is perfectly
        // inconsitent. Maybe the previous `&mut (*_2)` came from a different
        // MIR version. At this point there is no value in even checking.
        //
        // Looking at `temp_borrow_mixed_2` it seems like the copy mutability depends
        // on the use case. I'm not even disappointed anymore
        match bro_info.kind {
            BroKind::Dep | BroKind::Named => {
                // FIXME: Maybe this doesn't even needs to be tracked?
                self.borrows.insert(dst.local, bro_info.copy_with(BroKind::Dep));
            },
            // Only anons should be able to add new information
            BroKind::Anon => {
                let (is_all, is_part) = bro_info.borrowed_props();
                let is_named = matches!(info.locals[dst.local].kind, LocalKind::UserVar(..));
                if is_named {
                    // FIXME: THis is broken:
                    if matches!(bro_info.muta, Mutability::Mut) {
                        info.stats.borrow_mut().owned.named_borrow_mut_count += 1;
                        if is_all {
                            pats.insert(OwnedPat::NamedBorrow);
                        } else if is_part {
                            pats.insert(OwnedPat::PartNamedBorrow);
                        } else {
                            unreachable!();
                        }
                    } else {
                        info.stats.borrow_mut().owned.named_borrow_count += 1;

                        if is_all {
                            pats.insert(OwnedPat::NamedBorrowMut);
                        } else if is_part {
                            pats.insert(OwnedPat::PartNamedBorrowMut);
                        } else {
                            unreachable!();
                        }
                    }
                }

                let new_bro_kind = if is_named { BroKind::Named } else { BroKind::Anon };

                self.borrows.insert(dst.local, bro_info.copy_with(new_bro_kind));
            },
        }
    }

    fn update_bros(&mut self, broker: Place<'tcx>, muta: Mutability, info: &AnalysisInfo<'tcx>) {
        // I switch on muta before the `retain`, to make the `retain` specialized and therefore faster.
        match muta {
            // Not mutable aka aliasable
            Mutability::Not => self.borrows.retain(|_key, bro_info| {
                !(matches!(bro_info.muta, Mutability::Mut) && info.places_conflict(bro_info.broker, broker))
            }),
            Mutability::Mut => self
                .borrows
                .retain(|_key, bro_info| !info.places_conflict(bro_info.broker, broker)),
        }
    }

    pub fn has_bro(&self, anon: &Place<'_>) -> Option<BorrowInfo<'tcx>> {
        if let Some((_loc, place)) = self.phase_borrow.iter().find(|(local, _place)| *local == anon.local) {
            // TwoPhaseBorrows are always mutable
            Some(BorrowInfo::new(*place, Mutability::Mut, BroKind::Anon))
        } else {
            self.borrows.get(&anon.local).copied()
        }
    }
}

impl<'a, 'tcx> MyStateInfo<super::OwnedAnalysis<'a, 'tcx>> for StateInfo<'tcx> {
    fn new(bb: BasicBlock) -> Self {
        Self {
            bb,
            state: Default::default(),
            anons: Default::default(),
            borrows: Default::default(),
            phase_borrow: Default::default(),
            containers: Default::default(),
        }
    }

    fn join(&mut self, state_owner: &mut super::OwnedAnalysis<'a, 'tcx>, bb: BasicBlock) -> bool {
        let other = &state_owner.states[bb];
        if other.state.is_empty() {
            return false;
        }
        assert_ne!(other.state(), State::None);

        // Base case where `self` is uninit
        if self.state.is_empty() {
            let bb = self.bb;
            *self = other.clone();
            self.bb = bb;
            return true;
        }

        let self_state = self.state.last().copied().unwrap();
        let other_state = other.state.last().copied().unwrap();
        if self.state.len() != other.state.len() || self_state != other_state {
            // println!("- Merge:");
            // println!("    - {:?}", self.state);
            // println!("    - {:?}", other.state);
            let other_events = inspect_deviation(
                &self.state,
                &other.state,
                &mut state_owner.pats,
                |(base, _), deviation, pats| {
                    // println!("- Case 1 | 2:");
                    // println!("    - {base:?}");
                    // println!("    - {deviation:?}");
                    if matches!(base, State::Filled) {
                        let has_fill = deviation.iter().any(|(state, _)| matches!(state, State::Filled));
                        if has_fill {
                            pats.insert(OwnedPat::ConditionalOverwride);
                        }

                        let has_drop = deviation.iter().any(|(state, _)| matches!(state, State::Dropped));
                        if has_drop {
                            pats.insert(OwnedPat::ConditionalDrop);
                        }

                        let has_move = deviation.iter().any(|(state, _)| matches!(state, State::Moved));
                        if has_move {
                            pats.insert(OwnedPat::ConditionalMove);
                        }
                    }
                },
                |(base, _), a, b, pats| {
                    // println!("- Case 3:");
                    // println!("    - {base:?}");
                    // println!("    - {a:?}");
                    // println!("    - {b:?}");
                    if matches!(base, State::Empty) {
                        let a_fill = a.iter().any(|(state, _)| matches!(state, State::Filled));
                        let b_fill = b.iter().any(|(state, _)| matches!(state, State::Filled));

                        if a_fill || b_fill {
                            pats.insert(OwnedPat::ConditionalInit);
                        }
                    }
                },
            );
            self.state.extend(other_events.iter().copied());

            // TODO: Proper merging here
            let new_state = match (self.validity(), other.validity()) {
                (Validity::Valid, Validity::Valid) => State::Filled,
                (Validity::Not, Validity::Not) => State::Empty,
                (_, _) => State::MaybeFilled,
            };
            self.state.push((new_state, self.bb));
        }

        for (anon, other_places) in &other.anons {
            if let Some(self_places) = self.anons.get_mut(anon) {
                if self_places != other_places {
                    todo!();
                }
            } else {
                self.anons.insert(*anon, other_places.clone());
            }
        }

        // FIXME: Here we can have collisions where two anons reference different places... oh no...
        self.borrows.extend(other.borrows.iter());

        self.phase_borrow.extend(other.phase_borrow.iter());

        true
    }

    fn check_continue_diff_for_pats(&self, _state_owner: &mut super::OwnedAnalysis<'a, 'tcx>, _con_block: BasicBlock) {
        todo!();
    }
}

/// ```text
///     Case 1       Case 2          Case 3    //
///       x            x               x       //
///     / |            | \           /   \     //
///    *  |            |  *         *     *    //
///     \ |            | /           \   /     //
///       x            x               x       //
/// ```
/// This returns the deviation of the additional events from the b branch to be
/// added to the a collection for the next iteration.
fn inspect_deviation<'b>(
    a: &[(State, BasicBlock)],
    b: &'b [(State, BasicBlock)],
    pats: &mut BTreeSet<OwnedPat>,
    mut single_devitation: impl FnMut((State, BasicBlock), &[(State, BasicBlock)], &mut BTreeSet<OwnedPat>),
    mut split_devitation: impl FnMut(
        (State, BasicBlock),
        &[(State, BasicBlock)],
        &[(State, BasicBlock)],
        &mut BTreeSet<OwnedPat>,
    ),
) -> &'b [(State, BasicBlock)] {
    let a_state = a.last().copied().unwrap();
    let b_state = b.last().copied().unwrap();

    // Case 1
    if let Some(idx) = a.iter().rposition(|state| *state == b_state) {
        let base = a[idx];
        single_devitation(base, &a[(idx + 1)..], pats);
        return &[];
    }

    // Case 2
    if let Some(idx) = b.iter().rposition(|state| *state == a_state) {
        let base = b[idx];
        single_devitation(base, &b[(idx + 1)..], pats);
        return &b[(idx + 1)..];
    }

    let mut b_set = GrowableBitSet::with_capacity(a_state.1.as_usize().max(b_state.1.as_usize()) + 1);
    for (_, bb) in b {
        b_set.insert(*bb);
    }

    // Case 3
    if let Some((a_idx, &base)) = a.iter().enumerate().rev().find(|(_, (_, bb))| b_set.contains(*bb))
        && let Some(b_idx) = b.iter().rposition(|state| *state == base)
    {
        split_devitation(base, &a[(a_idx + 1)..], &b[(b_idx + 1)..], pats);
        return &b[(b_idx + 1)..];
    }

    unreachable!()
}
