//! This pass propagates upvars to locals within a generator, if it can
//! determine such coalescing is sound.
//!
//! Original demo: search for: `_3 = _1.0`; if found, then replace all occurrences of `_3` with `_1.0`.
//!
//! The generalization:
//!
//! * apply to more upvars than just `_1.0`; e.g. `_1.1`, `_1.2`, ... (but don't need to worry
//!   about array indices nor enum variants).
//!
//! * transitively apply the replacement to subsequent copies of the local(s).
//!   e.g. `_3 = _1.0; ... _4 = _3; ...; _5 = _4;`, then might replace `_5` with `_1.0`.
//!
//! * How do we know how long to make these "chains" of locals that are participating in these
//!   assignments? For now we use a simple answer: We extend them until we see a use of a local
//!   from the chain that *isn't* a trivial assignment to another local, and then cut off
//!   everything *after* that used local.
//!
//! * BUT, if a local is assigned with more than one root value (which we might approximate via
//!   more than one assignment), then we cannot include that local in the transformation.
//!
//! * Also, if `_3` occurs within a place that has a *Deref projection*, then we cannot do the
//!   replacement of `_3` with `_1.0` in that context (Post cleanup, all Deref projections must
//!   occur first within a place.). This means we cannot get rid of the initialization of `_3`, and
//!   thus probably should not do the replacement of `_3` with `_1.0` anywhere at all.
//!
//! * Furthermore, locals can occur in the projection itself, e.g. as `Index(_3)` in a PlaceElem.
//!   We cannot replace such locals with the `_1.0`, so this should likewise make the local
//!   ineligible.
//!
//! Notes around soundness:
//!
//! * any writes to a local with a rhs that isn't either the upvar or a local in that upvar's set
//!   should cause that local and all locals derived *from it* to become ineligible.
//!
//! * any writes to `_1` should invalidate the whole transformation. (Should be at most a corner
//!   case.)
//!
//! * any write to `_1.L` where `.L` is one of the upvars that we are using as root values should
//!   invalidate the use of that upvar as a root value. e.g. a write to `_1.2` should make `_1.2`
//!   ineligible for the transformation, but *other* upvars like `_1.0` should still be able to
//!   participate.
//!
//! But, what about transmutes of `&_1` and writes to fields of the transmuted thing? We simplify
//! the reasoning here by just treating *any* use of `_1` that isn't a trivial assignment of a
//! field to a local (e.g. `&mut _1`, `&_1`) as invalidating the whole transformation

use crate::MirPass;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_target::abi::FieldIdx;

struct IndexMap<K: Idx, V> {
    vec: IndexVec<K, Option<V>>,
}

impl<K: Idx, V: Copy> IndexMap<K, V> {
    fn new(n: usize) -> Self {
        IndexMap { vec: IndexVec::from_elem_n(None, n) }
    }
    fn get(&self, k: K) -> Option<V> {
        self.vec.get(k).map(|p| *p).flatten()
    }
}

const SELF_IDX: u32 = 1;
const SELF_ARG: Local = Local::from_u32(SELF_IDX);

pub struct UpvarToLocalProp;

// This is a way to make `trace!` format strings shorter to placate tidy.
const W: &'static str = "UpvarToLocalProp";

struct Patch<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Maps each local to upvar that transformation will use as local's replacement
    local_to_root_upvar_and_ty: IndexMap<Local, (FieldIdx, Ty<'tcx>)>,
}

impl<'tcx> MirPass<'tcx> for UpvarToLocalProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0 // on by default w/o -Zmir-opt-level=0
    }
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running UpvarToLocalProp on {:?}", body.source);

        // First: how many upvars might we need to consider?
        let upvar_limit: usize = {
            // If we are looking at a generator, we have a fast path
            if body.yield_ty().is_some() {
                // The first argument is the generator type passed by value
                let gen_ty = body.local_decls.raw[SELF_IDX as usize].ty;
                if let ty::Generator(_, substs, _movability) = gen_ty.kind() {
                    let substs = substs.as_generator();
                    substs.upvar_tys().len()
                } else {
                    tcx.sess.delay_span_bug(
                        body.span,
                        &format!("unexpected generator type {}", gen_ty),
                    );
                    return;
                }
            } else {
                // otherwise, fall back to scanning code to find "upvar" count.
                struct MaxUpvarField(usize);
                impl<'tcx> Visitor<'tcx> for MaxUpvarField {
                    fn visit_place(
                        &mut self,
                        place: &Place<'tcx>,
                        _context: PlaceContext,
                        _location: Location,
                    ) {
                        if place.local == SELF_ARG {
                            if let Some(ProjectionElem::Field(field, _ty)) =
                                place.projection.as_slice().get(0)
                            {
                                self.0 = std::cmp::max(self.0, field.index());
                            }
                        }
                    }
                }
                let mut v = MaxUpvarField(0);
                v.visit_body(body);
                v.0 + 1
            }
        };

        let num_locals = body.local_decls.len();
        let mir_source = body.source.def_id();

        let upvar_to_locals = IndexVec::from_elem_n(Chain::new(), upvar_limit);
        let upvar_to_ty = IndexMap { vec: IndexVec::from_elem_n(None, upvar_limit) };
        let mut local_to_upvar = IndexVec::from_elem_n(LocalState::Unseen, num_locals);

        // The special return place `_0` must not participate in transformation.
        local_to_upvar[RETURN_PLACE] = LocalState::Ineligible;

        let mut walk = Walk { tcx, mir_source, upvar_to_locals, local_to_upvar, upvar_to_ty };

        walk.visit_body(body);
        let mut p = walk.to_patch();
        p.visit_body(body);
    }
}

/// Each upvar `_1.K` maps to a *chain* of locals [_l0, ... _li, ...] such that we see see `_l0 =
/// _1.K; ...; _l{i+1} = _li; ...` in the analyzed MIR, with no intervening invalidations, such
/// that it would be sound to make them an *alias-set* and replace all uses of the locals `_li`
/// with `_1.K` itself
///
/// During the analysis, this chain first grows (as we discover new copies), then shrinks
/// (potentially to nothing) as we discover uses of the locals that invalidate their inclusion in
/// that upvar's alias-set.
#[derive(Clone, Debug)]
struct Chain {
    /// The [_li ...] in the chain. Note that teh last element of the chain may have a special role
    /// depending on the ChainState `self.state`.
    local_copies: Vec<Local>,

    /// The `state` captures where we are in the analysis for the upvar: We may be still collecting
    /// all the copies, or we may have found the end of the copy chain and are now searching for
    /// reasons to trim it shorter.
    state: ChainState,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ChainState {
    /// A chain that is growing; i.e. we may discover another copy `_z = _y;` where _y is the last
    /// element of the vector, and can push _z onto the vector to mark its addition to the chain.
    Growing,

    /// We found a chain of copies, but we cannot grow it further because either:
    /// * the last element of the chain is used somewhere that isn't a local-to-local copy (but
    ///   is a place where it would be valid to substitute an upvar), such as a borrow, or
    /// * the last element of the chain is only used in a local-to-local copy, but the target of
    ///   that copy is used in a place where it would not be valid to substitute an upvar.
    Shrinking,

    /// Chain for associated upvar was entirely invalidated (e.g. by borrow of, write to, or
    /// another read from the upvar).
    Invalidated,
    // (One could try representing invalidated chains as Shrinking chains of zero length, since a
    //  chain will only transition from growing to shrinking, never the other way. It is not clear
    //  whether such a representation would clarify or obfuscate the code here.)
}

impl Chain {
    fn new() -> Self {
        Chain { local_copies: Vec::new(), state: ChainState::Growing }
    }
    fn len(&self) -> usize {
        let len = self.local_copies.len();
        if len > 0 {
            assert_ne!(self.state, ChainState::Invalidated);
        }
        len
    }
    fn tail(&self) -> Option<Local> {
        let len = self.local_copies.len();
        if len > 0 { self.local_copies.get(len - 1).map(|l| *l) } else { None }
    }
    fn push_if_growing(&mut self, l: Local) -> bool {
        if self.state == ChainState::Growing {
            self.local_copies.push(l);
            true
        } else {
            false
        }
    }
    fn stop_growing(&mut self) {
        if self.state == ChainState::Growing {
            self.state = ChainState::Shrinking;
        }
    }
}

struct Walk<'tcx> {
    tcx: TyCtxt<'tcx>,

    mir_source: DefId,

    /// maps each upvar `_1.K` to a chain of locals [_l, ... _i, _j, ...] such
    /// that we see see `_l = _1.K; ...; _i = _; _j = _i; ...` in the analyzed
    /// MIR, with no intervening invalidations. The docs for `Chain` spell out
    /// the associated state transitions for the analysis.
    upvar_to_locals: IndexVec<FieldIdx, Chain>,

    /// maps each local to its associated upvar, if its still eligible for its
    /// single chain of copies.
    local_to_upvar: IndexVec<Local, LocalState>,

    /// The MIR is going to require we supply the Ty of the upvar when we
    /// substitute it for part of a Place.
    upvar_to_ty: IndexMap<FieldIdx, Ty<'tcx>>,
}

/// Reason that a invalidation step is taken.
#[derive(Copy, Clone)]
enum Reason<'tcx> {
    NonFieldSelfProjection,
    UseOfUpvar(FieldIdx),
    Write2ToLocal(Local),
    Read2FromUpvar(FieldIdx),
    DereferenceInPlace(Place<'tcx>),
    WriteToTail(Place<'tcx>, Location),
    WriteToPlace(Place<'tcx>),
    LocalUsedAsIndex(Local),
    ChainAlreadyExtendedPast(Local),
}

impl<'tcx> std::fmt::Display for Reason<'tcx> {
    fn fmt(&self, w: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reason::NonFieldSelfProjection => {
                write!(w, "observed non-field projection on self; giving up on transform.")
            }
            Reason::Write2ToLocal(l) => {
                write!(w, "observed 2nd write to local {l:?}; it and its suffix now invalidated.")
            }
            Reason::Read2FromUpvar(f) => {
                write!(w, "observed 2nd read from upvar {f:?}; it is now invalidated.")
            }
            Reason::UseOfUpvar(f) => {
                write!(w, "observed a non-trivial use of upvar _1.{f:?}; it is now ineligible.")
            }
            Reason::DereferenceInPlace(place) => {
                let local = &place.local;
                write!(w, "ineligible local:{local:?} because of dereference in {place:?}")
            }
            Reason::WriteToTail(place, loc) => {
                let local = &place.local;
                let Location { block: bb, statement_index: idx } = loc;
                let bb = bb.index();
                write!(
                    w,
                    "observed write to place:{place:?} at bb{bb}:{idx}; {local:?} now tail at most."
                )
            }
            Reason::WriteToPlace(place) => {
                let local = &place.local;
                write!(w, "observed write to place:{place:?}; {local:?} is now ineligible.")
            }
            Reason::LocalUsedAsIndex(local) => {
                write!(w, "observed local:{local:?} used as index; it is now ineligible.")
            }
            Reason::ChainAlreadyExtendedPast(local) => {
                write!(w, "local:{local:?} is not the tail of its chain")
            }
        }
    }
}

impl<'tcx> Walk<'tcx> {
    fn num_locals(&self) -> usize {
        self.local_to_upvar.len()
    }

    // case: `_l = _r;`
    fn saw_local_to_local(&mut self, lhs: Local, rhs: Local) {
        match self.local_to_upvar[lhs] {
            LocalState::Unseen => {
                // fall through to happy path where we might grow the chain.
            }
            LocalState::Valid(_) | LocalState::Ineligible => {
                // not the first write to lhs; we must ensure lhs is invalidated, and *also* must
                // immediately end growth of chain for rhs.
                self.invalidate_local(lhs, Reason::Write2ToLocal(lhs));
                self.stop_growing_chain_for_local(rhs);

                // (still fall-through in case rhs needs further invalidation below.)
            }
        }

        // if rhs is valid chain, then must inspect further; otherwise, no more invalidation to do.
        let LocalState::Valid(rhs_upvar) = self.local_to_upvar[rhs] else { return; };

        let rhs_chain = &mut self.upvar_to_locals[rhs_upvar];
        assert!(rhs_chain.local_copies.contains(&rhs));
        if rhs_chain.tail() != Some(rhs) {
            // Uh oh, this chain has *already* been extended with a different assignment from rhs.
            // That means we must cut the chain and force rhs to be the end of it.
            self.cut_off_only_beyond_local(rhs, Reason::ChainAlreadyExtendedPast(rhs));
            return;
        }

        assert_eq!(self.local_to_upvar[rhs], LocalState::Valid(rhs_upvar));
        assert_eq!(rhs_chain.tail(), Some(rhs));

        if self.local_to_upvar[lhs] == LocalState::Unseen {
            self.try_to_alias(lhs, rhs_upvar);
        }
    }

    // case: `_l = _1.field;`
    fn saw_upvar_to_local(&mut self, lhs: Local, (rhs, rhs_ty): (FieldIdx, Ty<'tcx>)) {
        let chains = &self.upvar_to_locals;
        let locals = &self.local_to_upvar;

        let needs_invalidation = {
            if let LocalState::Valid(_) = self.local_to_upvar[lhs] {
                // if lhs L is already in some alias-set, then this is a new write to L and we must
                // trim that alias set back now.
                Some(Reason::Write2ToLocal(lhs))
            } else if chains[rhs].len() > 0 {
                // If we already had entries, then we have two uses of the same upvar for `rhs`,
                // and must give up on it entirely.
                Some(Reason::Read2FromUpvar(rhs))
            } else {
                None
            }
        };

        if let Some(reason) = needs_invalidation {
            self.invalidate_local(lhs, reason);
            self.invalidate_upvar(rhs, reason);
            return;
        }

        if locals[lhs] != LocalState::Unseen {
            return;
        }

        // At this point, (1.) we have confirmed that (a.) this is the first time we have seen lhs,
        // and (b.) the chain for rhs is length zero (but not necessarily growing) and (2.) we have
        // committed to *try* to make lhs and rhs aliases.

        if self.upvar_to_ty.vec[rhs] == None {
            self.upvar_to_ty.vec[rhs] = Some(rhs_ty);
        }
        assert_eq!(self.upvar_to_ty.vec[rhs], Some(rhs_ty));

        self.try_to_alias(lhs, rhs);
    }

    fn try_to_alias(&mut self, lhs: Local, rhs: FieldIdx) {
        let locals = &mut self.local_to_upvar;
        assert_eq!(locals[lhs], LocalState::Unseen);

        // If optimization fuel exhausted, then do not add more entries to any alias set.
        if !self.tcx.consider_optimizing(|| format!("UpvarToLocalProp {:?}", self.mir_source)) {
            return;
        }
        if self.upvar_to_locals[rhs].push_if_growing(lhs) {
            locals[lhs] = LocalState::Valid(rhs);
        } else {
            locals[lhs] = LocalState::Ineligible;
        }
    }

    fn invalidate_all(&mut self, reason: Reason<'tcx>) {
        trace!("{W} invalidate_all, {reason}");
        for chain in self.upvar_to_locals.iter_mut() {
            chain.local_copies.clear();
            chain.state = ChainState::Invalidated;
        }
        for local_state in self.local_to_upvar.iter_mut() {
            *local_state = LocalState::Ineligible;
        }
    }

    fn invalidate_upvar(&mut self, f: FieldIdx, reason: Reason<'tcx>) {
        trace!("{W} invalidate_upvar, {reason}");
        let chain = &mut self.upvar_to_locals[f];
        while let Some(l) = chain.local_copies.pop() {
            self.local_to_upvar[l] = LocalState::Ineligible;
        }
        chain.state = ChainState::Invalidated;
    }

    fn invalidate_local(&mut self, l: Local, reason: Reason<'tcx>) {
        trace!("{W} invalidate_local, {reason}");
        match self.local_to_upvar[l] {
            LocalState::Valid(_) => self.cut_off_local_and_beyond(l, reason),
            LocalState::Unseen => self.local_to_upvar[l] = LocalState::Ineligible,
            LocalState::Ineligible => {}
        }
    }

    fn cut_off_only_beyond_local(&mut self, l: Local, reason: Reason<'tcx>) {
        trace!("{W} cut_off_only_beyond_local, {reason}");
        self.cut_off_suffix(l, true);
    }

    fn cut_off_local_and_beyond(&mut self, l: Local, reason: Reason<'tcx>) {
        trace!("{W} cut_off_local_and_beyond, {reason}");
        self.cut_off_suffix(l, false);
    }

    fn cut_off_suffix(&mut self, l: Local, keep_local: bool) {
        let locals = &mut self.local_to_upvar;
        let LocalState::Valid(upvar) = locals[l] else {
            panic!("should only be called for locals currently attached to an upvar chain");
        };

        let chain = &mut self.upvar_to_locals[upvar];
        assert!(chain.local_copies.contains(&l));
        chain.stop_growing();

        while let Some(tail) = chain.local_copies.pop() {
            assert_eq!(locals[tail], LocalState::Valid(upvar));
            if keep_local && tail == l {
                chain.local_copies.push(tail);
                break;
            }

            locals[tail] = LocalState::Ineligible;

            if tail == l {
                assert!(!keep_local);
                break;
            }
        }
    }

    fn stop_growing_chain_for_local(&mut self, l: Local) {
        if let LocalState::Valid(upvar) = self.local_to_upvar[l] {
            self.upvar_to_locals[upvar].stop_growing();
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LocalState {
    /// Unseen: We have not yet seen this local in the walk over the MIR
    Unseen,
    /// Valid(F): We have seen this local exactly once, and there is a direct chain of copies
    /// `_l0 = _self.F; _l1 = _l0; ...; L = _li;` that connects this local L to F.
    Valid(FieldIdx),
    /// This local cannot be part of any alias set; furthermore, if it is the left-hand side of any
    /// local-to-local assignment, then the right-hand side of that assignment is either (a.) not
    /// in a chain of copies, or (b.) must be *the* tail of its chain of copies.
    Ineligible,
}

impl<'tcx> Visitor<'tcx> for Walk<'tcx> {
    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        match categorize_assign(place, rvalue) {
            AssignCategory::LocalToLocal { lhs, rhs } => self.saw_local_to_local(lhs, rhs),
            AssignCategory::UpvarToLocal { lhs, rhs: (rhs, rhs_ty) } => {
                self.saw_upvar_to_local(lhs, (rhs, rhs_ty))
            }
            AssignCategory::Other => {
                // recurs if *and only if* we are in the other "complex" cases.
                self.super_assign(place, rvalue, location)
            }
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        let local = place.local;
        // If there is a Deref within place, then we cannot do the replacement in that place, and
        // thus should not bother trying to replace that local at all.
        if place.is_indirect() {
            self.invalidate_local(local, Reason::DereferenceInPlace(*place));
        } else if context.is_use() {
            if place.local == SELF_ARG {
                // case: use of `_1`
                if let Some(ProjectionElem::Field(f, _ty)) = place.projection.get(0) {
                    self.invalidate_upvar(*f, Reason::UseOfUpvar(*f));
                } else {
                    // for *every* other kind of projection of _1, we will just invalidate the
                    // *whole* transformation.
                    self.invalidate_all(Reason::NonFieldSelfProjection);
                }
            } else {
                // case: use of some place derived from a local other than `_1`.
                if context.is_drop() {
                    // FIXME: make sure I can justify this, especially in face of conditional
                    // control flow. Plus, the drop-independence may be a property that holds for
                    // the upvars, but not for the locals being put in alias set for that upvar.
                    trace!("{W} drop of place:{place:?} does not affect analysis state");
                } else if let LocalState::Valid(_) = self.local_to_upvar[local] {
                    // A local `L` is allowed to be used (e.g. `&L`, `&mut L`, or `L.foo = 3`) as
                    // the very last link in a chain of aliases, but cannot be the source of an
                    // assignment at that point.
                    let reason = Reason::WriteToTail(*place, location);
                    self.cut_off_only_beyond_local(place.local, reason);
                } else {
                    // If we see a write to `L` *before* it is on any chain, then do not allow
                    // it to be part of any chain in the future.
                    self.invalidate_local(local, Reason::WriteToPlace(*place));
                }
            }
        }
        self.super_place(place, context, location)
    }

    // A fun corner case: you can have MIR like: `_18 = &(*_27)[_3];`, and you cannot plug `_1.0`
    // in for `_3` in that context.
    fn visit_projection_elem(
        &mut self,
        _place_ref: PlaceRef<'tcx>,
        elem: PlaceElem<'tcx>,
        _context: PlaceContext,
        _location: Location,
    ) {
        match elem {
            ProjectionElem::OpaqueCast(_ty) | ProjectionElem::Field(_, _ty) => {}

            ProjectionElem::Deref
            | ProjectionElem::Subslice { from: _, to: _, from_end: _ }
            | ProjectionElem::ConstantIndex { offset: _, min_length: _, from_end: _ }
            | ProjectionElem::Downcast(_, _) => {}

            ProjectionElem::Index(local) => {
                self.invalidate_local(local, Reason::LocalUsedAsIndex(local));
            }
        }
    }
}

impl<'tcx> Walk<'tcx> {
    fn to_patch(self) -> Patch<'tcx> {
        let tcx = self.tcx;
        Patch { tcx, local_to_root_upvar_and_ty: self.to_local_map() }
    }

    fn to_local_map(self) -> IndexMap<Local, (FieldIdx, Ty<'tcx>)> {
        let mut map = IndexMap::new(self.num_locals());

        // This assertion is not strictly necessary, in the sense that the analysis and
        // transformation are still sound if we only zip over a prefix of these sequences.
        //
        // (Also note that these vectors include invalidated chains and uninitialized types, so
        // this isn't really asserting all that much at all...)
        assert_eq!(self.upvar_to_locals.len(), self.upvar_to_ty.vec.len());

        let chains = self.upvar_to_locals.iter_enumerated();
        let chains_and_tys = chains.zip(self.upvar_to_ty.vec.into_iter());
        for ((field, chain), ty) in chains_and_tys {
            let Some(ty) = ty else { assert_eq!(chain.len(), 0); continue; };
            if chain.state == ChainState::Invalidated {
                assert_eq!(chain.local_copies.len(), 0);
            }
            for local in &chain.local_copies {
                assert!(map.vec[*local].is_none());
                map.vec[*local] = Some((field, ty));
            }
        }

        map
    }
}

enum AssignCategory<'tcx> {
    /// case of `_lhs = _rhs;`
    LocalToLocal { lhs: Local, rhs: Local },
    /// case of `_lhs = self.field;` aka `_lhs = _1.field;`
    UpvarToLocal { lhs: Local, rhs: (FieldIdx, Ty<'tcx>) },
    /// every other kind of assignment
    Other,
}

fn categorize_assign<'tcx>(place: &Place<'tcx>, rvalue: &Rvalue<'tcx>) -> AssignCategory<'tcx> {
    let l = place.local;
    if place.projection.len() == 0 &&
        let Rvalue::Use(Operand::Copy(rhs)) | Rvalue::Use(Operand::Move(rhs)) =
        rvalue
    {
        match (rhs.local, rhs.projection.as_slice()) {
            // case `_l = _r;`
            (r, &[]) => AssignCategory::LocalToLocal { lhs: l, rhs: r },
            // case `_l = _1.field;`
            (r, &[ProjectionElem::Field(f, ty)]) if r == SELF_ARG => {
                AssignCategory::UpvarToLocal { lhs: l, rhs: (f, ty) }
            }
            _ => AssignCategory::Other,
        }
    } else {
        AssignCategory::Other
    }
}

impl<'tcx> MutVisitor<'tcx> for Patch<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _context: PlaceContext, _location: Location) {
        // Any point in the MIR that has a local that is eligible for replacement should be *doing*
        // that replacement, which means *this* `visit_local` method should never ever see any such
        // local.
        //
        // As a safeguard (because pnkfelix overlooked one such case, namely
        // `ProjectionElem::Index(Local)`), assert that here.
        //
        // (Note however, this safeguard is imperfect and won't catch all coding errors, because
        // for this check to even occur, we need the right recursive calls to `super_place` and
        // `super_statement` to remain in the right places.)
        if let Some((upvar, ty)) = self.local_to_root_upvar_and_ty.get(*local) {
            panic!(
                "{W} visited local:{local:?} at {_context:?} and found \
                 unexpected associated upvar:{upvar:?} of type:{ty:?}"
            );
        }
    }

    fn visit_place(
        &mut self,
        place: &mut Place<'tcx>,
        _context: PlaceContext,
        _location: Location,
    ) {
        if let Some(Some((field, ty))) = self.local_to_root_upvar_and_ty.vec.get(place.local) {
            let new_base = Place {
                local: SELF_ARG,
                projection: self.tcx.mk_place_elems(&[ProjectionElem::Field(*field, *ty)]),
            };
            trace!("{W} replacing base of {place:?} with new base {new_base:?}");
            crate::replace_base(place, new_base, self.tcx);
        }
        self.super_place(place, _context, _location);
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = statement.kind {
            if let Some(Some(_)) = self.local_to_root_upvar_and_ty.vec.get(l) {
                trace!("{W} replacing newly useless storage marker {statement:?} with no-op.");
                statement.kind = StatementKind::Nop;
                return;
            }
        }

        let mut inspect_for_no_op_swap = false;
        if let StatementKind::Assign(tup) = &statement.kind {
            let lhs = &tup.0;
            if let Some(Some(_)) = self.local_to_root_upvar_and_ty.vec.get(lhs.local) {
                inspect_for_no_op_swap = true;
            }
        }
        self.super_statement(statement, location);
        if inspect_for_no_op_swap {
            let StatementKind::Assign(tup) = &statement.kind else {
                panic!("cannot lose assign during super_statement call");
            };
            let lhs = &tup.0;
            let rhs = &tup.1;
            match rhs {
                Rvalue::Use(Operand::Copy(p)) | Rvalue::Use(Operand::Move(p)) if p == lhs => {
                    trace!("{W} replacing rewritten {statement:?} with no-op.");
                    statement.kind = StatementKind::Nop;
                }
                _ => {}
            }
        }
    }
}
