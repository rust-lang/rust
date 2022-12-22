use super::possible_origin::PossibleOriginVisitor;
use crate::ty::is_copy;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_index::bit_set::{BitSet, HybridBitSet};
use rustc_lint::LateContext;
use rustc_middle::mir::{
    self, visit::Visitor as _, BasicBlock, Local, Location, Mutability, Statement, StatementKind, Terminator,
};
use rustc_middle::ty::{self, visit::TypeVisitor, TyCtxt};
use rustc_mir_dataflow::{
    fmt::DebugWithContext, impls::MaybeStorageLive, lattice::JoinSemiLattice, Analysis, AnalysisDomain,
    CallReturnPlaces, ResultsCursor,
};
use std::borrow::Cow;
use std::ops::ControlFlow;

/// Collects the possible borrowers of each local.
/// For example, `b = &a; c = &a;` will make `b` and (transitively) `c`
/// possible borrowers of `a`.
#[allow(clippy::module_name_repetitions)]
struct PossibleBorrowerAnalysis<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'b mir::Body<'tcx>,
    possible_origin: FxHashMap<mir::Local, HybridBitSet<mir::Local>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PossibleBorrowerState {
    map: FxIndexMap<Local, BitSet<Local>>,
    domain_size: usize,
}

impl PossibleBorrowerState {
    fn new(domain_size: usize) -> Self {
        Self {
            map: FxIndexMap::default(),
            domain_size,
        }
    }

    #[allow(clippy::similar_names)]
    fn add(&mut self, borrowed: Local, borrower: Local) {
        self.map
            .entry(borrowed)
            .or_insert(BitSet::new_empty(self.domain_size))
            .insert(borrower);
    }
}

impl<C> DebugWithContext<C> for PossibleBorrowerState {
    fn fmt_with(&self, _ctxt: &C, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <_ as std::fmt::Debug>::fmt(self, f)
    }
    fn fmt_diff_with(&self, _old: &Self, _ctxt: &C, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

impl JoinSemiLattice for PossibleBorrowerState {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (&borrowed, borrowers) in other.map.iter() {
            if !borrowers.is_empty() {
                changed |= self
                    .map
                    .entry(borrowed)
                    .or_insert(BitSet::new_empty(self.domain_size))
                    .union(borrowers);
            }
        }
        changed
    }
}

impl<'b, 'tcx> AnalysisDomain<'tcx> for PossibleBorrowerAnalysis<'b, 'tcx> {
    type Domain = PossibleBorrowerState;

    const NAME: &'static str = "possible_borrower";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        PossibleBorrowerState::new(body.local_decls.len())
    }

    fn initialize_start_block(&self, _body: &mir::Body<'tcx>, _entry_set: &mut Self::Domain) {}
}

impl<'b, 'tcx> PossibleBorrowerAnalysis<'b, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        body: &'b mir::Body<'tcx>,
        possible_origin: FxHashMap<mir::Local, HybridBitSet<mir::Local>>,
    ) -> Self {
        Self {
            tcx,
            body,
            possible_origin,
        }
    }
}

impl<'b, 'tcx> Analysis<'tcx> for PossibleBorrowerAnalysis<'b, 'tcx> {
    fn apply_call_return_effect(
        &self,
        _state: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }

    fn apply_statement_effect(&self, state: &mut Self::Domain, statement: &Statement<'tcx>, _location: Location) {
        if let StatementKind::Assign(box (place, rvalue)) = &statement.kind {
            let lhs = place.local;
            match rvalue {
                mir::Rvalue::Ref(_, _, borrowed) => {
                    state.add(borrowed.local, lhs);
                },
                other => {
                    if ContainsRegion
                        .visit_ty(place.ty(&self.body.local_decls, self.tcx).ty)
                        .is_continue()
                    {
                        return;
                    }
                    rvalue_locals(other, |rhs| {
                        if lhs != rhs {
                            state.add(rhs, lhs);
                        }
                    });
                },
            }
        }
    }

    fn apply_terminator_effect(&self, state: &mut Self::Domain, terminator: &Terminator<'tcx>, _location: Location) {
        if let mir::TerminatorKind::Call {
            args,
            destination: mir::Place { local: dest, .. },
            ..
        } = &terminator.kind
        {
            // TODO add doc
            // If the call returns something with lifetimes,
            // let's conservatively assume the returned value contains lifetime of all the arguments.
            // For example, given `let y: Foo<'a> = foo(x)`, `y` is considered to be a possible borrower of `x`.

            let mut immutable_borrowers = vec![];
            let mut mutable_borrowers = vec![];

            for op in args {
                match op {
                    mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                        if let ty::Ref(_, _, Mutability::Mut) = self.body.local_decls[p.local].ty.kind() {
                            mutable_borrowers.push(p.local);
                        } else {
                            immutable_borrowers.push(p.local);
                        }
                    },
                    mir::Operand::Constant(..) => (),
                }
            }

            let mut mutable_variables: Vec<mir::Local> = mutable_borrowers
                .iter()
                .filter_map(|r| self.possible_origin.get(r))
                .flat_map(HybridBitSet::iter)
                .collect();

            if ContainsRegion.visit_ty(self.body.local_decls[*dest].ty).is_break() {
                mutable_variables.push(*dest);
            }

            for y in mutable_variables {
                for x in &immutable_borrowers {
                    state.add(*x, y);
                }
                for x in &mutable_borrowers {
                    state.add(*x, y);
                }
            }
        }
    }
}

struct ContainsRegion;

impl TypeVisitor<'_> for ContainsRegion {
    type BreakTy = ();

    fn visit_region(&mut self, _: ty::Region<'_>) -> ControlFlow<Self::BreakTy> {
        ControlFlow::BREAK
    }
}

fn rvalue_locals(rvalue: &mir::Rvalue<'_>, mut visit: impl FnMut(mir::Local)) {
    use rustc_middle::mir::Rvalue::{Aggregate, BinaryOp, Cast, CheckedBinaryOp, Repeat, UnaryOp, Use};

    let mut visit_op = |op: &mir::Operand<'_>| match op {
        mir::Operand::Copy(p) | mir::Operand::Move(p) => visit(p.local),
        mir::Operand::Constant(..) => (),
    };

    match rvalue {
        Use(op) | Repeat(op, _) | Cast(_, op, _) | UnaryOp(_, op) => visit_op(op),
        Aggregate(_, ops) => ops.iter().for_each(visit_op),
        BinaryOp(_, box (lhs, rhs)) | CheckedBinaryOp(_, box (lhs, rhs)) => {
            visit_op(lhs);
            visit_op(rhs);
        },
        _ => (),
    }
}

/// Result of `PossibleBorrowerAnalysis`.
#[allow(clippy::module_name_repetitions)]
pub struct PossibleBorrowerMap<'b, 'tcx> {
    body: &'b mir::Body<'tcx>,
    possible_borrower: ResultsCursor<'b, 'tcx, PossibleBorrowerAnalysis<'b, 'tcx>>,
    maybe_live: ResultsCursor<'b, 'tcx, MaybeStorageLive<'b>>,
    pushed: BitSet<Local>,
    stack: Vec<Local>,
}

impl<'b, 'tcx> PossibleBorrowerMap<'b, 'tcx> {
    pub fn new(cx: &LateContext<'tcx>, mir: &'b mir::Body<'tcx>) -> Self {
        let possible_origin = {
            let mut vis = PossibleOriginVisitor::new(mir);
            vis.visit_body(mir);
            vis.into_map(cx)
        };
        let possible_borrower = PossibleBorrowerAnalysis::new(cx.tcx, mir, possible_origin)
            .into_engine(cx.tcx, mir)
            .pass_name("possible_borrower")
            .iterate_to_fixpoint()
            .into_results_cursor(mir);
        let maybe_live = MaybeStorageLive::new(Cow::Owned(BitSet::new_empty(mir.local_decls.len())))
            .into_engine(cx.tcx, mir)
            .pass_name("possible_borrower")
            .iterate_to_fixpoint()
            .into_results_cursor(mir);
        PossibleBorrowerMap {
            body: mir,
            possible_borrower,
            maybe_live,
            pushed: BitSet::new_empty(mir.local_decls.len()),
            stack: Vec::with_capacity(mir.local_decls.len()),
        }
    }

    /// Returns true if the set of borrowers of `borrowed` living at `at` includes no more than
    /// `borrowers`.
    /// Notes:
    /// 1. It would be nice if `PossibleBorrowerMap` could store `cx` so that `at_most_borrowers`
    /// would not require it to be passed in. But a `PossibleBorrowerMap` is stored in `LintPass`
    /// `Dereferencing`, which outlives any `LateContext`.
    /// 2. In all current uses of `at_most_borrowers`, `borrowers` is a slice of at most two
    /// elements. Thus, `borrowers.contains(...)` is effectively a constant-time operation. If
    /// `at_most_borrowers`'s uses were to expand beyond this, its implementation might have to be
    /// adjusted.
    pub fn at_most_borrowers(
        &mut self,
        cx: &LateContext<'tcx>,
        borrowers: &[mir::Local],
        borrowed: mir::Local,
        at: mir::Location,
    ) -> bool {
        if is_copy(cx, self.body.local_decls[borrowed].ty) {
            return true;
        }

        self.possible_borrower.seek_before_primary_effect(at);
        self.maybe_live.seek_before_primary_effect(at);

        let possible_borrower = &self.possible_borrower.get().map;
        let maybe_live = &self.maybe_live;

        self.pushed.clear();
        self.stack.clear();

        if let Some(borrowers) = possible_borrower.get(&borrowed) {
            for b in borrowers.iter() {
                if self.pushed.insert(b) {
                    self.stack.push(b);
                }
            }
        } else {
            // Nothing borrows `borrowed` at `at`.
            return true;
        }

        while let Some(borrower) = self.stack.pop() {
            if maybe_live.contains(borrower) && !borrowers.contains(&borrower) {
                return false;
            }

            if let Some(borrowers) = possible_borrower.get(&borrower) {
                for b in borrowers.iter() {
                    if self.pushed.insert(b) {
                        self.stack.push(b);
                    }
                }
            }
        }

        true
    }

    pub fn local_is_alive_at(&mut self, local: mir::Local, at: mir::Location) -> bool {
        self.maybe_live.seek_after_primary_effect(at);
        self.maybe_live.contains(local)
    }
}
