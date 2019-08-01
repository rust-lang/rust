use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::visit::{MutatingUseContext, NonMutatingUseContext, NonUseContext};
use rustc::mir::{self, Local, Location, Place, PlaceBase};
use rustc::ty::{self, TyCtxt, ParamEnv};
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_target::spec::abi::Abi;

use smallvec::SmallVec;
use syntax::symbol::sym;

use super::{BitDenotation, GenKillSet, BottomValue};

newtype_index! {
    /// A linear index for each definition in the program.
    ///
    /// This must be linear since it will be used to index a bit vector containing the set of
    /// reaching definitions during dataflow analysis.
    pub struct DefIndex {
        DEBUG_FORMAT = "def{}",
    }
}

/// A dataflow analysis which tracks whether an assignment to a variable might still constitute the
/// value of that variable at a given point in the program.
///
/// Reaching definitions uses the term "definition" to refer to places in a program where a
/// variable's value may change. Most definitions are writes to a variable (e.g. via
/// `StatementKind::Assign`). However, a variable can also be implicitly defined by calling code if
/// it is a function parameter or defined to be uninitialized via a call to
/// `MaybeUninit::uninitialized`.
///
/// An example, using parentheses to denote each definition.
///
/// ```rust,no_run
/// fn collatz_step(mut n: i32) -> i32 { // (1) Implicit definiton of function param
///     if n % 2 == 0 {
///         n /= 2                       // (2)
///     } else {
///         n = 3*n + 1                  // (3)
///     }
///
///     // Only definitions (2) and (3) could possibly constitute the value of `n` at function exit.
///     // (1), the `n` which was passed into the function, is killed by either (2) or (3)
///     // along all paths to this point.
///     n
/// }
/// ```
///
/// Not all programs are so simple; some care is required to handle projections. For field and
/// array index projections, we cannot kill all previous definitions of the same variable, since
/// only part of the variable was defined. If we encounter a deref projection on the left-hand side
/// of an assignment, we don't even know *which* variable is being defined, since the pointer being
/// dereferenced may be pointing anywhere. Such definitions are never killed. Finally, we must
/// treat function calls as opaque: We can't know (without MIR inlining or explicit annotation)
/// whether a callee mutates data behind a pointer. If the address of one of our locals is
/// observable, it may be the target of such a mutation. A type-based analysis (e.g. does this
/// function take any type containing a mutable reference as a parameter?) is insufficient, since
/// raw pointers can be laundered through any integral type.
///
/// At the moment, the possible targets of a definition are tracked at the granularity of a
/// `Local`, not a `MovePath` as is done in the initialized places analysis. That means that a
/// definition like `x.y = 1` does not kill a previous assignment of `x.y`. This could be made more
/// precise with more work; it is not a fundamental limitation of the analysis.
///
/// Most passes will want to know which definitions of a **specific** local reach a given point in
/// the program. `ReachingDefinitions` does not provide this information; it can only provide all
/// definitions of **any** local. However, a `ReachingDefinitions` analysis can be used to build a
/// `UseDefChain`, which does provide this information.
pub struct ReachingDefinitions {
    all: IndexVec<DefIndex, Definition>,

    /// All (direct) definitions of a local in the program.
    for_local_direct: IndexVec<Local, Vec<DefIndex>>,

    /// All definitions at the given location in the program.
    at_location: FxHashMap<Location, SmallVec<[DefIndex; 4]>>,
}

fn has_side_effects(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    param_env: ParamEnv<'tcx>,
    terminator: &mir::Terminator<'tcx>,
) -> bool {
    match &terminator.kind {
        mir::TerminatorKind::Call { .. } => true,

        // Types with special drop glue may mutate their environment.
        | mir::TerminatorKind::Drop { location: place, .. }
        | mir::TerminatorKind::DropAndReplace { location: place, .. }
        => place.ty(body, tcx).ty.needs_drop(tcx, param_env),

        | mir::TerminatorKind::Goto { .. }
        | mir::TerminatorKind::SwitchInt { .. }
        | mir::TerminatorKind::Resume
        | mir::TerminatorKind::Abort
        | mir::TerminatorKind::Return
        | mir::TerminatorKind::Unreachable
        | mir::TerminatorKind::Assert { .. }
        | mir::TerminatorKind::FalseEdges { .. }
        | mir::TerminatorKind::FalseUnwind { .. }
        => false,

        // FIXME: I don't know the semantics around these so assume that they may mutate their
        // environment.
        | mir::TerminatorKind::Yield { .. }
        | mir::TerminatorKind::GeneratorDrop
        => true,
    }
}

impl ReachingDefinitions {
    pub fn new<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, param_env: ParamEnv<'tcx>) -> Self {
        let mut ret = ReachingDefinitions {
            all: IndexVec::new(),
            for_local_direct: IndexVec::from_elem(Vec::new(), &body.local_decls),
            at_location: Default::default(),
        };

        let mut builder = DefBuilder { defs: &mut ret };
        for arg in body.args_iter() {
            builder.add_arg_def(arg);
        }

        DefVisitor(|place: &Place<'_>, loc| builder.visit_def(place, loc))
            .visit_body(body);

        // Add all function calls as indirect definitions.
        let blocks_with_side_effects = body
            .basic_blocks()
            .iter_enumerated()
            .filter(|(_, data)| has_side_effects(tcx, body, param_env, data.terminator()));

        for (block, data) in blocks_with_side_effects {
            let term_loc = Location { block, statement_index: data.statements.len() };
            builder.add_def(term_loc, DefKind::Indirect);
        }

        ret
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn get(&self, idx: DefIndex) -> &Definition {
        &self.all[idx]
    }

    /// Iterates over all definitions which go through a layer of indirection (e.g. `(*_1) = ...`).
    pub fn indirect(&self) -> impl Iterator<Item = DefIndex> + '_ {
        self.all
            .iter_enumerated()
            .filter(|(_, def)| def.is_indirect())
            .map(|(id, _)| id)
    }

    pub fn for_local_direct(&self, local: Local) -> impl Iterator<Item = DefIndex> + '_ {
        self.for_local_direct[local].iter().cloned()
    }

    pub fn at_location(&self, location: Location) -> impl Iterator<Item = DefIndex> + '_ {
        self.at_location
            .get(&location)
            .into_iter()
            .flat_map(|v| v.iter().cloned())
    }

    pub fn for_args(&self) -> impl Iterator<Item = DefIndex> + '_ {
        self.all
            .iter_enumerated()
            .take_while(|(_, def)| def.location.is_none())
            .map(|(def, _)| def)
    }

    fn update_trans(&self, trans: &mut GenKillSet<DefIndex>, location: Location) {
        for def in self.at_location(location) {
            trans.gen(def);

            // If we assign directly to a local (e.g. `_1 = ...`), we can kill all other
            // direct definitions of that local. We must not kill indirect definitions, since
            // they may define locals other than the one currently being assigned to.
            //
            // FIXME: If we assign to a field of a local `_1.x = ...`, we could kill all other
            // definitions of any part of that field (e.g. `_1.x.y = ...`). This would require
            // tracking `MovePath`s instead of `Local`s.
            if let DefKind::DirectWhole(local) = self.get(def).kind {
                let other_direct_defs = self
                    .for_local_direct(local)
                    .filter(|&d| d != def);

                trans.kill_all(other_direct_defs);
            }
        }
    }
}

impl<'tcx> BitDenotation<'tcx> for ReachingDefinitions {
    type Idx = DefIndex;
    fn name() -> &'static str { "reaching_definitions" }

    fn bits_per_block(&self) -> usize {
        self.len()
    }

    fn start_block_effect(&self, entry_set: &mut BitSet<Self::Idx>) {
        // Our parameters were defined by the caller at function entry.
        for def in self.for_args() {
            entry_set.insert(def);
        }
    }

    fn statement_effect(
        &self,
        trans: &mut GenKillSet<Self::Idx>,
        location: Location,
    ) {
        self.update_trans(trans, location);
    }

    fn terminator_effect(
        &self,
        trans: &mut GenKillSet<Self::Idx>,
        location: Location,
    ) {
        self.update_trans(trans, location);
    }

    fn propagate_call_return(
        &self,
        _in_out: &mut BitSet<Self::Idx>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        _dest_place: &mir::Place<'tcx>,
    ) {
        // FIXME: RETURN_PLACE should only be defined when a call returns successfully.
    }
}

impl BottomValue for ReachingDefinitions {
    /// bottom = definition not reachable
    const BOTTOM_VALUE: bool = false;
}

struct DefBuilder<'a> {
    defs: &'a mut ReachingDefinitions,
}

impl DefBuilder<'_> {
    fn add_arg_def(&mut self, arg: Local) -> DefIndex {
        let def = Definition { kind: DefKind::DirectWhole(arg), location: None };
        let idx = self.defs.all.push(def);

        self.defs.for_local_direct[arg].push(idx);
        idx
    }

    fn add_def(&mut self, location: Location, kind: DefKind<Local>) -> DefIndex {
        let def = Definition { kind, location: Some(location) };
        let idx = self.defs.all.push(def);

        if let Some(local) = kind.direct() {
            self.defs.for_local_direct[local].push(idx);
        }

        self.defs.at_location.entry(location).or_default().push(idx);
        idx
    }

    fn visit_def(&mut self, place: &mir::Place<'tcx>, location: Location) {
        // We don't track reaching defs for statics.
        let def_for_local = DefKind::of_place(place)
            .map(PlaceBase::local)
            .transpose();

        if let Some(def) = def_for_local {
            self.add_def(location, def);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Definition {
    pub kind: DefKind<Local>,

    /// The `Location` at which this definition occurs, or `None` if it occurs before the
    /// `mir::Body` is entered (e.g. the value of an argument).
    pub location: Option<Location>,
}

impl Definition {
    fn is_indirect(&self) -> bool {
        self.kind == DefKind::Indirect
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind<T> {
    /// An entire variable is defined directly, e.g. `_1 = ...`.
    DirectWhole(T),

    /// Some part of a variable (a field or an array offset) is defined directly, e.g. `_1.field =
    /// ...` or `_1[2] = ...`.
    DirectPart(T),

    /// An unknown variable is defined through a pointer, e.g. `(*_1) = ...`.
    ///
    /// In general, we cannot know where `_1` is pointing: it may reference any local in the
    /// current `Body` (whose address has been observed) or somewhere else entirely (e.g. a static,
    /// the heap, or a local further down the stack).
    Indirect,
}

impl<T> DefKind<T> {
    /// Returns `Some(T)` if the `DefKind` is direct or `None` if it is indirect.
    pub fn direct(self) -> Option<T> {
        match self {
            DefKind::DirectWhole(x) | DefKind::DirectPart(x) => Some(x),
            DefKind::Indirect => None,
        }
    }

    /// Converts a `DefKind<T>` to a `DefKind<U>` by applying a function to the value contained in
    /// either `DirectWhole` or `DirectPart`.
    fn map<U>(self, f: impl FnOnce(T) -> U) -> DefKind<U> {
        match self {
            DefKind::DirectWhole(x) => DefKind::DirectWhole(f(x)),
            DefKind::DirectPart(x) => DefKind::DirectPart(f(x)),
            DefKind::Indirect => DefKind::Indirect,
        }
    }
}

impl<T> DefKind<Option<T>> {
    /// Converts a `DefKind` of an `Option` into an `Option` of a `DefKind`.
    ///
    /// The result is `None` if either direct variant contains `None`. Otherwise, the result is
    /// `Some(self.map(Option::unwrap))`.
    fn transpose(self) -> Option<DefKind<T>> {
        match self {
            DefKind::DirectWhole(Some(x)) => Some(DefKind::DirectWhole(x)),
            DefKind::DirectPart(Some(x)) => Some(DefKind::DirectPart(x)),
            DefKind::Indirect => Some(DefKind::Indirect),

            DefKind::DirectWhole(None) | DefKind::DirectPart(None) => None,
        }
    }
}

impl<'a, 'tcx> DefKind<&'a mir::PlaceBase<'tcx>> {
    fn of_place(place: &'a mir::Place<'tcx>) -> Self {
        place.iterate(|base, projections| {
            let mut is_whole = true;
            for proj in projections {
                is_whole = false;
                if proj.elem == mir::ProjectionElem::Deref {
                    return DefKind::Indirect;
                }
            }

            if is_whole {
                DefKind::DirectWhole(base)
            } else {
                DefKind::DirectPart(base)
            }
        })
    }
}

pub struct DefVisitor<F>(pub F);

impl<'tcx, F> Visitor<'tcx> for DefVisitor<F>
    where F: FnMut(&Place<'tcx>, Location)
{
    fn visit_place(&mut self,
                   place: &Place<'tcx>,
                   context: PlaceContext,
                   location: Location) {
        self.super_place(place, context, location);

        match context {
            // DEFS
            | PlaceContext::MutatingUse(MutatingUseContext::AsmOutput)
            | PlaceContext::MutatingUse(MutatingUseContext::Call)
            | PlaceContext::MutatingUse(MutatingUseContext::Store)
            => self.0(place, location),

            | PlaceContext::MutatingUse(MutatingUseContext::Borrow)
            | PlaceContext::MutatingUse(MutatingUseContext::Drop)
            | PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::MutatingUse(MutatingUseContext::Retag)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::ShallowBorrow)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::UniqueBorrow)
            | PlaceContext::NonUse(NonUseContext::AscribeUserTy)
            | PlaceContext::NonUse(NonUseContext::StorageDead)
            | PlaceContext::NonUse(NonUseContext::StorageLive)
            => (),
        }
    }
}

pub struct UseVisitor<F>(pub F);

impl<'tcx, F> Visitor<'tcx> for UseVisitor<F>
    where F: FnMut(&Place<'tcx>, Location)
{
    fn visit_place(&mut self,
                   place: &Place<'tcx>,
                   context: PlaceContext,
                   location: Location) {
        self.super_place(place, context, location);

        match context {
            | PlaceContext::MutatingUse(MutatingUseContext::AsmOutput)
            | PlaceContext::MutatingUse(MutatingUseContext::Borrow)
            | PlaceContext::MutatingUse(MutatingUseContext::Drop)
            | PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::ShallowBorrow)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::UniqueBorrow)
            => self.0(place, location),

            | PlaceContext::MutatingUse(MutatingUseContext::Call)
            | PlaceContext::MutatingUse(MutatingUseContext::Retag)
            | PlaceContext::MutatingUse(MutatingUseContext::Store)
            | PlaceContext::NonUse(NonUseContext::AscribeUserTy)
            | PlaceContext::NonUse(NonUseContext::StorageDead)
            | PlaceContext::NonUse(NonUseContext::StorageLive)
            => (),
        }
    }
}
