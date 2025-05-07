use super::possible_origin::PossibleOriginVisitor;
use super::transitive_relation::TransitiveRelation;
use crate::ty::is_copy;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_lint::LateContext;
use rustc_middle::mir::visit::Visitor as _;
use rustc_middle::mir::{self, Mutability};
use rustc_middle::ty::{self, TyCtxt, TypeVisitor};
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use std::borrow::Cow;
use std::ops::ControlFlow;

/// Collects the possible borrowers of each local.
/// For example, `b = &a; c = &a;` will make `b` and (transitively) `c`
/// possible borrowers of `a`.
#[allow(clippy::module_name_repetitions)]
struct PossibleBorrowerVisitor<'a, 'b, 'tcx> {
    possible_borrower: TransitiveRelation,
    body: &'b mir::Body<'tcx>,
    cx: &'a LateContext<'tcx>,
    possible_origin: FxHashMap<mir::Local, DenseBitSet<mir::Local>>,
}

impl<'a, 'b, 'tcx> PossibleBorrowerVisitor<'a, 'b, 'tcx> {
    fn new(
        cx: &'a LateContext<'tcx>,
        body: &'b mir::Body<'tcx>,
        possible_origin: FxHashMap<mir::Local, DenseBitSet<mir::Local>>,
    ) -> Self {
        Self {
            possible_borrower: TransitiveRelation::default(),
            body,
            cx,
            possible_origin,
        }
    }

    fn into_map(
        self,
        cx: &'a LateContext<'tcx>,
        maybe_live: ResultsCursor<'b, 'tcx, MaybeStorageLive<'tcx>>,
    ) -> PossibleBorrowerMap<'b, 'tcx> {
        let mut map = FxHashMap::default();
        for row in (1..self.body.local_decls.len()).map(mir::Local::from_usize) {
            if is_copy(cx, self.body.local_decls[row].ty) {
                continue;
            }

            let mut borrowers = self.possible_borrower.reachable_from(row, self.body.local_decls.len());
            borrowers.remove(mir::Local::from_usize(0));
            if !borrowers.is_empty() {
                map.insert(row, borrowers);
            }
        }

        let bs = DenseBitSet::new_empty(self.body.local_decls.len());
        PossibleBorrowerMap {
            map,
            maybe_live,
            bitset: (bs.clone(), bs),
        }
    }
}

impl<'tcx> mir::visit::Visitor<'tcx> for PossibleBorrowerVisitor<'_, '_, 'tcx> {
    fn visit_assign(&mut self, place: &mir::Place<'tcx>, rvalue: &mir::Rvalue<'_>, _location: mir::Location) {
        let lhs = place.local;
        match rvalue {
            mir::Rvalue::Ref(_, _, borrowed) | mir::Rvalue::CopyForDeref(borrowed) => {
                self.possible_borrower.add(borrowed.local, lhs);
            },
            other => {
                if ContainsRegion
                    .visit_ty(place.ty(&self.body.local_decls, self.cx.tcx).ty)
                    .is_continue()
                {
                    return;
                }
                rvalue_locals(other, |rhs| {
                    if lhs != rhs {
                        self.possible_borrower.add(rhs, lhs);
                    }
                });
            },
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'_>, _loc: mir::Location) {
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
                match &op.node {
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
                .flat_map(DenseBitSet::iter)
                .collect();

            if ContainsRegion.visit_ty(self.body.local_decls[*dest].ty).is_break() {
                mutable_variables.push(*dest);
            }

            for y in mutable_variables {
                for x in &immutable_borrowers {
                    self.possible_borrower.add(*x, y);
                }
                for x in &mutable_borrowers {
                    self.possible_borrower.add(*x, y);
                }
            }
        }
    }
}

struct ContainsRegion;

impl TypeVisitor<TyCtxt<'_>> for ContainsRegion {
    type Result = ControlFlow<()>;

    fn visit_region(&mut self, _: ty::Region<'_>) -> Self::Result {
        ControlFlow::Break(())
    }
}

fn rvalue_locals(rvalue: &mir::Rvalue<'_>, mut visit: impl FnMut(mir::Local)) {
    use rustc_middle::mir::Rvalue::{Aggregate, BinaryOp, Cast, Repeat, UnaryOp, Use};

    let mut visit_op = |op: &mir::Operand<'_>| match op {
        mir::Operand::Copy(p) | mir::Operand::Move(p) => visit(p.local),
        mir::Operand::Constant(..) => (),
    };

    match rvalue {
        Use(op) | Repeat(op, _) | Cast(_, op, _) | UnaryOp(_, op) => visit_op(op),
        Aggregate(_, ops) => ops.iter().for_each(visit_op),
        BinaryOp(_, box (lhs, rhs)) => {
            visit_op(lhs);
            visit_op(rhs);
        },
        _ => (),
    }
}

/// Result of `PossibleBorrowerVisitor`.
#[allow(clippy::module_name_repetitions)]
pub struct PossibleBorrowerMap<'b, 'tcx> {
    /// Mapping `Local -> its possible borrowers`
    pub map: FxHashMap<mir::Local, DenseBitSet<mir::Local>>,
    maybe_live: ResultsCursor<'b, 'tcx, MaybeStorageLive<'tcx>>,
    // Caches to avoid allocation of `DenseBitSet` on every query
    pub bitset: (DenseBitSet<mir::Local>, DenseBitSet<mir::Local>),
}

impl<'b, 'tcx> PossibleBorrowerMap<'b, 'tcx> {
    pub fn new(cx: &LateContext<'tcx>, mir: &'b mir::Body<'tcx>) -> Self {
        let possible_origin = {
            let mut vis = PossibleOriginVisitor::new(mir);
            vis.visit_body(mir);
            vis.into_map(cx)
        };
        let maybe_storage_live_result =
            MaybeStorageLive::new(Cow::Owned(DenseBitSet::new_empty(mir.local_decls.len())))
                .iterate_to_fixpoint(cx.tcx, mir, Some("redundant_clone"))
                .into_results_cursor(mir);
        let mut vis = PossibleBorrowerVisitor::new(cx, mir, possible_origin);
        vis.visit_body(mir);
        vis.into_map(cx, maybe_storage_live_result)
    }

    /// Returns true if the set of borrowers of `borrowed` living at `at` matches with `borrowers`.
    pub fn only_borrowers(&mut self, borrowers: &[mir::Local], borrowed: mir::Local, at: mir::Location) -> bool {
        self.bounded_borrowers(borrowers, borrowers, borrowed, at)
    }

    /// Returns true if the set of borrowers of `borrowed` living at `at` includes at least `below`
    /// but no more than `above`.
    pub fn bounded_borrowers(
        &mut self,
        below: &[mir::Local],
        above: &[mir::Local],
        borrowed: mir::Local,
        at: mir::Location,
    ) -> bool {
        self.maybe_live.seek_after_primary_effect(at);

        self.bitset.0.clear();
        let maybe_live = &mut self.maybe_live;
        if let Some(bitset) = self.map.get(&borrowed) {
            for b in bitset.iter().filter(move |b| maybe_live.get().contains(*b)) {
                self.bitset.0.insert(b);
            }
        } else {
            return false;
        }

        self.bitset.1.clear();
        for b in below {
            self.bitset.1.insert(*b);
        }

        if !self.bitset.0.superset(&self.bitset.1) {
            return false;
        }

        for b in above {
            self.bitset.0.remove(*b);
        }

        self.bitset.0.is_empty()
    }

    pub fn local_is_alive_at(&mut self, local: mir::Local, at: mir::Location) -> bool {
        self.maybe_live.seek_after_primary_effect(at);
        self.maybe_live.get().contains(local)
    }
}
