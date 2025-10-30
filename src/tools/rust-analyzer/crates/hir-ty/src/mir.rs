//! MIR definitions and implementation

use std::{collections::hash_map::Entry, fmt::Display, iter};

use base_db::Crate;
use either::Either;
use hir_def::{
    DefWithBodyId, FieldId, StaticId, TupleFieldId, UnionId, VariantId,
    expr_store::Body,
    hir::{BindingAnnotation, BindingId, Expr, ExprId, Ordering, PatId},
};
use la_arena::{Arena, ArenaMap, Idx, RawIdx};
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashMap;
use rustc_type_ir::inherent::{AdtDef, GenericArgs as _, IntoKind, SliceLike, Ty as _};
use smallvec::{SmallVec, smallvec};
use stdx::{impl_from, never};

use crate::{
    CallableDefId, InferenceResult, MemoryMap,
    consteval::usize_const,
    db::{HirDatabase, InternedClosureId},
    display::{DisplayTarget, HirDisplay},
    infer::PointerCast,
    lang_items::is_box,
    next_solver::{
        Const, DbInterner, ErrorGuaranteed, GenericArgs, ParamEnv, Ty, TyKind,
        infer::{InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

mod borrowck;
mod eval;
mod lower;
mod monomorphization;
mod pretty;

pub use borrowck::{BorrowckResult, MutabilityReason, borrowck_query};
pub use eval::{
    Evaluator, MirEvalError, VTableMap, interpret_mir, pad16, render_const_using_debug_impl,
};
pub use lower::{MirLowerError, lower_to_mir, mir_body_for_closure_query, mir_body_query};
pub use monomorphization::{
    monomorphized_mir_body_for_closure_query, monomorphized_mir_body_query,
};

pub(crate) use lower::mir_body_cycle_result;
pub(crate) use monomorphization::monomorphized_mir_body_cycle_result;

use super::consteval::try_const_usize;

pub type BasicBlockId<'db> = Idx<BasicBlock<'db>>;
pub type LocalId<'db> = Idx<Local<'db>>;

fn return_slot<'db>() -> LocalId<'db> {
    LocalId::from_raw(RawIdx::from(0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local<'db> {
    pub ty: Ty<'db>,
}

/// An operand in MIR represents a "value" in Rust, the definition of which is undecided and part of
/// the memory model. One proposal for a definition of values can be found [on UCG][value-def].
///
/// [value-def]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/value-domain.md
///
/// The most common way to create values is via loading a place. Loading a place is an operation
/// which reads the memory of the place and converts it to a value. This is a fundamentally *typed*
/// operation. The nature of the value produced depends on the type of the conversion. Furthermore,
/// there may be other effects: if the type has a validity constraint loading the place might be UB
/// if the validity constraint is not met.
///
/// **Needs clarification:** Ralf proposes that loading a place not have side-effects.
/// This is what is implemented in miri today. Are these the semantics we want for MIR? Is this
/// something we can even decide without knowing more about Rust's memory model?
///
/// **Needs clarification:** Is loading a place that has its variant index set well-formed? Miri
/// currently implements it, but it seems like this may be something to check against in the
/// validator.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Operand<'db> {
    kind: OperandKind<'db>,
    // FIXME : This should actually just be of type `MirSpan`.
    span: Option<MirSpan>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum OperandKind<'db> {
    /// Creates a value by loading the given place.
    ///
    /// Before drop elaboration, the type of the place must be `Copy`. After drop elaboration there
    /// is no such requirement.
    Copy(Place<'db>),

    /// Creates a value by performing loading the place, just like the `Copy` operand.
    ///
    /// This *may* additionally overwrite the place with `uninit` bytes, depending on how we decide
    /// in [UCG#188]. You should not emit MIR that may attempt a subsequent second load of this
    /// place without first re-initializing it.
    ///
    /// [UCG#188]: https://github.com/rust-lang/unsafe-code-guidelines/issues/188
    Move(Place<'db>),
    /// Constants are already semantically values, and remain unchanged.
    Constant { konst: Const<'db>, ty: Ty<'db> },
    /// NON STANDARD: This kind of operand returns an immutable reference to that static memory. Rustc
    /// handles it with the `Constant` variant somehow.
    Static(StaticId),
}

impl<'db> Operand<'db> {
    fn from_concrete_const(data: Box<[u8]>, memory_map: MemoryMap<'db>, ty: Ty<'db>) -> Self {
        let interner = DbInterner::conjure();
        Operand {
            kind: OperandKind::Constant {
                konst: Const::new_valtree(interner, ty, data, memory_map),
                ty,
            },
            span: None,
        }
    }

    fn from_bytes(data: Box<[u8]>, ty: Ty<'db>) -> Self {
        Operand::from_concrete_const(data, MemoryMap::default(), ty)
    }

    fn const_zst(ty: Ty<'db>) -> Operand<'db> {
        Self::from_bytes(Box::default(), ty)
    }

    fn from_fn(
        db: &'db dyn HirDatabase,
        func_id: hir_def::FunctionId,
        generic_args: GenericArgs<'db>,
    ) -> Operand<'db> {
        let interner = DbInterner::new_with(db, None, None);
        let ty = Ty::new_fn_def(interner, CallableDefId::FunctionId(func_id).into(), generic_args);
        Operand::from_bytes(Box::default(), ty)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProjectionElem<V, T> {
    Deref,
    Field(Either<FieldId, TupleFieldId>),
    // FIXME: get rid of this, and use FieldId for tuples and closures
    ClosureField(usize),
    Index(V),
    ConstantIndex { offset: u64, from_end: bool },
    Subslice { from: u64, to: u64 },
    //Downcast(Option<Symbol>, VariantIdx),
    OpaqueCast(T),
}

impl<V, T> ProjectionElem<V, T> {
    pub fn projected_ty<'db>(
        &self,
        infcx: &InferCtxt<'db>,
        mut base: Ty<'db>,
        closure_field: impl FnOnce(InternedClosureId, GenericArgs<'db>, usize) -> Ty<'db>,
        krate: Crate,
    ) -> Ty<'db> {
        let interner = infcx.interner;
        let db = interner.db;

        // we only bail on mir building when there are type mismatches
        // but error types may pop up resulting in us still attempting to build the mir
        // so just propagate the error type
        if base.is_ty_error() {
            return Ty::new_error(interner, ErrorGuaranteed);
        }

        if matches!(base.kind(), TyKind::Alias(..)) {
            let mut ocx = ObligationCtxt::new(infcx);
            // FIXME: we should get this from caller
            let env = ParamEnv::empty();
            match ocx.structurally_normalize_ty(&ObligationCause::dummy(), env, base) {
                Ok(it) => base = it,
                Err(_) => return Ty::new_error(interner, ErrorGuaranteed),
            }
        }

        match self {
            ProjectionElem::Deref => match base.kind() {
                TyKind::RawPtr(inner, _) | TyKind::Ref(_, inner, _) => inner,
                TyKind::Adt(adt_def, subst) if is_box(db, adt_def.def_id().0) => subst.type_at(0),
                _ => {
                    never!(
                        "Overloaded deref on type {} is not a projection",
                        base.display(db, DisplayTarget::from_crate(db, krate))
                    );
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            ProjectionElem::Field(Either::Left(f)) => match base.kind() {
                TyKind::Adt(_, subst) => {
                    db.field_types(f.parent)[f.local_id].instantiate(interner, subst)
                }
                ty => {
                    never!("Only adt has field, found {:?}", ty);
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            ProjectionElem::Field(Either::Right(f)) => match base.kind() {
                TyKind::Tuple(subst) => {
                    subst.as_slice().get(f.index as usize).copied().unwrap_or_else(|| {
                        never!("Out of bound tuple field");
                        Ty::new_error(interner, ErrorGuaranteed)
                    })
                }
                ty => {
                    never!("Only tuple has tuple field: {:?}", ty);
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            ProjectionElem::ClosureField(f) => match base.kind() {
                TyKind::Closure(id, subst) => closure_field(id.0, subst, *f),
                _ => {
                    never!("Only closure has closure field");
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            ProjectionElem::ConstantIndex { .. } | ProjectionElem::Index(_) => match base.kind() {
                TyKind::Array(inner, _) | TyKind::Slice(inner) => inner,
                _ => {
                    never!("Overloaded index is not a projection");
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            &ProjectionElem::Subslice { from, to } => match base.kind() {
                TyKind::Array(inner, c) => {
                    let next_c = usize_const(
                        db,
                        match try_const_usize(db, c) {
                            None => None,
                            Some(x) => x.checked_sub(u128::from(from + to)),
                        },
                        krate,
                    );
                    Ty::new_array_with_const_len(interner, inner, next_c)
                }
                TyKind::Slice(_) => base,
                _ => {
                    never!("Subslice projection should only happen on slice and array");
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            ProjectionElem::OpaqueCast(_) => {
                never!("We don't emit these yet");
                Ty::new_error(interner, ErrorGuaranteed)
            }
        }
    }
}

type PlaceElem<'db> = ProjectionElem<LocalId<'db>, Ty<'db>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectionId(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionStore<'db> {
    id_to_proj: FxHashMap<ProjectionId, Box<[PlaceElem<'db>]>>,
    proj_to_id: FxHashMap<Box<[PlaceElem<'db>]>, ProjectionId>,
}

impl Default for ProjectionStore<'_> {
    fn default() -> Self {
        let mut this = Self { id_to_proj: Default::default(), proj_to_id: Default::default() };
        // Ensure that [] will get the id 0 which is used in `ProjectionId::Empty`
        this.intern(Box::new([]));
        this
    }
}

impl<'db> ProjectionStore<'db> {
    pub fn shrink_to_fit(&mut self) {
        self.id_to_proj.shrink_to_fit();
        self.proj_to_id.shrink_to_fit();
    }

    pub fn intern_if_exist(&self, projection: &[PlaceElem<'db>]) -> Option<ProjectionId> {
        self.proj_to_id.get(projection).copied()
    }

    pub fn intern(&mut self, projection: Box<[PlaceElem<'db>]>) -> ProjectionId {
        let new_id = ProjectionId(self.proj_to_id.len() as u32);
        match self.proj_to_id.entry(projection) {
            Entry::Occupied(id) => *id.get(),
            Entry::Vacant(e) => {
                let key_clone = e.key().clone();
                e.insert(new_id);
                self.id_to_proj.insert(new_id, key_clone);
                new_id
            }
        }
    }
}

impl ProjectionId {
    pub const EMPTY: ProjectionId = ProjectionId(0);

    pub fn is_empty(self) -> bool {
        self == ProjectionId::EMPTY
    }

    pub fn lookup<'a, 'db>(self, store: &'a ProjectionStore<'db>) -> &'a [PlaceElem<'db>] {
        store.id_to_proj.get(&self).unwrap()
    }

    pub fn project<'db>(
        self,
        projection: PlaceElem<'db>,
        store: &mut ProjectionStore<'db>,
    ) -> ProjectionId {
        let mut current = self.lookup(store).to_vec();
        current.push(projection);
        store.intern(current.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Place<'db> {
    pub local: LocalId<'db>,
    pub projection: ProjectionId,
}

impl<'db> Place<'db> {
    fn is_parent(&self, child: &Place<'db>, store: &ProjectionStore<'db>) -> bool {
        self.local == child.local
            && child.projection.lookup(store).starts_with(self.projection.lookup(store))
    }

    /// The place itself is not included
    fn iterate_over_parents<'a>(
        &'a self,
        store: &'a ProjectionStore<'db>,
    ) -> impl Iterator<Item = Place<'db>> + 'a {
        let projection = self.projection.lookup(store);
        (0..projection.len()).map(|x| &projection[0..x]).filter_map(move |x| {
            Some(Place { local: self.local, projection: store.intern_if_exist(x)? })
        })
    }

    fn project(&self, projection: PlaceElem<'db>, store: &mut ProjectionStore<'db>) -> Place<'db> {
        Place { local: self.local, projection: self.projection.project(projection, store) }
    }
}

impl<'db> From<LocalId<'db>> for Place<'db> {
    fn from(local: LocalId<'db>) -> Self {
        Self { local, projection: ProjectionId::EMPTY }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AggregateKind<'db> {
    /// The type is of the element
    Array(Ty<'db>),
    /// The type is of the tuple
    Tuple(Ty<'db>),
    Adt(VariantId, GenericArgs<'db>),
    Union(UnionId, FieldId),
    Closure(Ty<'db>),
    //Coroutine(LocalDefId, SubstsRef, Movability),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SwitchTargets<'db> {
    /// Possible values. The locations to branch to in each case
    /// are found in the corresponding indices from the `targets` vector.
    values: SmallVec<[u128; 1]>,

    /// Possible branch sites. The last element of this vector is used
    /// for the otherwise branch, so targets.len() == values.len() + 1
    /// should hold.
    //
    // This invariant is quite non-obvious and also could be improved.
    // One way to make this invariant is to have something like this instead:
    //
    // branches: Vec<(ConstInt, BasicBlock)>,
    // otherwise: Option<BasicBlock> // exhaustive if None
    //
    // However we’ve decided to keep this as-is until we figure a case
    // where some other approach seems to be strictly better than other.
    targets: SmallVec<[BasicBlockId<'db>; 2]>,
}

impl<'db> SwitchTargets<'db> {
    /// Creates switch targets from an iterator of values and target blocks.
    ///
    /// The iterator may be empty, in which case the `SwitchInt` instruction is equivalent to
    /// `goto otherwise;`.
    pub fn new(
        targets: impl Iterator<Item = (u128, BasicBlockId<'db>)>,
        otherwise: BasicBlockId<'db>,
    ) -> Self {
        let (values, mut targets): (SmallVec<_>, SmallVec<_>) = targets.unzip();
        targets.push(otherwise);
        Self { values, targets }
    }

    /// Builds a switch targets definition that jumps to `then` if the tested value equals `value`,
    /// and to `else_` if not.
    pub fn static_if(value: u128, then: BasicBlockId<'db>, else_: BasicBlockId<'db>) -> Self {
        Self { values: smallvec![value], targets: smallvec![then, else_] }
    }

    /// Returns the fallback target that is jumped to when none of the values match the operand.
    pub fn otherwise(&self) -> BasicBlockId<'db> {
        *self.targets.last().unwrap()
    }

    /// Returns an iterator over the switch targets.
    ///
    /// The iterator will yield tuples containing the value and corresponding target to jump to, not
    /// including the `otherwise` fallback target.
    ///
    /// Note that this may yield 0 elements. Only the `otherwise` branch is mandatory.
    pub fn iter(&self) -> impl Iterator<Item = (u128, BasicBlockId<'db>)> + '_ {
        iter::zip(&self.values, &self.targets).map(|(x, y)| (*x, *y))
    }

    /// Returns a slice with all possible jump targets (including the fallback target).
    pub fn all_targets(&self) -> &[BasicBlockId<'db>] {
        &self.targets
    }

    /// Finds the `BasicBlock` to which this `SwitchInt` will branch given the
    /// specific value. This cannot fail, as it'll return the `otherwise`
    /// branch if there's not a specific match for the value.
    pub fn target_for_value(&self, value: u128) -> BasicBlockId<'db> {
        self.iter().find_map(|(v, t)| (v == value).then_some(t)).unwrap_or_else(|| self.otherwise())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Terminator<'db> {
    pub span: MirSpan,
    pub kind: TerminatorKind<'db>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TerminatorKind<'db> {
    /// Block has one successor; we continue execution there.
    Goto { target: BasicBlockId<'db> },

    /// Switches based on the computed value.
    ///
    /// First, evaluates the `discr` operand. The type of the operand must be a signed or unsigned
    /// integer, char, or bool, and must match the given type. Then, if the list of switch targets
    /// contains the computed value, continues execution at the associated basic block. Otherwise,
    /// continues execution at the "otherwise" basic block.
    ///
    /// Target values may not appear more than once.
    SwitchInt {
        /// The discriminant value being tested.
        discr: Operand<'db>,

        targets: SwitchTargets<'db>,
    },

    /// Indicates that the landing pad is finished and that the process should continue unwinding.
    ///
    /// Like a return, this marks the end of this invocation of the function.
    ///
    /// Only permitted in cleanup blocks. `Resume` is not permitted with `-C unwind=abort` after
    /// deaggregation runs.
    UnwindResume,

    /// Indicates that the landing pad is finished and that the process should abort.
    ///
    /// Used to prevent unwinding for foreign items or with `-C unwind=abort`. Only permitted in
    /// cleanup blocks.
    Abort,

    /// Returns from the function.
    ///
    /// Like function calls, the exact semantics of returns in Rust are unclear. Returning very
    /// likely at least assigns the value currently in the return place (`_0`) to the place
    /// specified in the associated `Call` terminator in the calling function, as if assigned via
    /// `dest = move _0`. It might additionally do other things, like have side-effects in the
    /// aliasing model.
    ///
    /// If the body is a coroutine body, this has slightly different semantics; it instead causes a
    /// `CoroutineState::Returned(_0)` to be created (as if by an `Aggregate` rvalue) and assigned
    /// to the return place.
    Return,

    /// Indicates a terminator that can never be reached.
    ///
    /// Executing this terminator is UB.
    Unreachable,

    /// The behavior of this statement differs significantly before and after drop elaboration.
    /// After drop elaboration, `Drop` executes the drop glue for the specified place, after which
    /// it continues execution/unwinds at the given basic blocks. It is possible that executing drop
    /// glue is special - this would be part of Rust's memory model. (**FIXME**: due we have an
    /// issue tracking if drop glue has any interesting semantics in addition to those of a function
    /// call?)
    ///
    /// `Drop` before drop elaboration is a *conditional* execution of the drop glue. Specifically, the
    /// `Drop` will be executed if...
    ///
    /// **Needs clarification**: End of that sentence. This in effect should document the exact
    /// behavior of drop elaboration. The following sounds vaguely right, but I'm not quite sure:
    ///
    /// > The drop glue is executed if, among all statements executed within this `Body`, an assignment to
    /// > the place or one of its "parents" occurred more recently than a move out of it. This does not
    /// > consider indirect assignments.
    Drop { place: Place<'db>, target: BasicBlockId<'db>, unwind: Option<BasicBlockId<'db>> },

    /// Drops the place and assigns a new value to it.
    ///
    /// This first performs the exact same operation as the pre drop-elaboration `Drop` terminator;
    /// it then additionally assigns the `value` to the `place` as if by an assignment statement.
    /// This assignment occurs both in the unwind and the regular code paths. The semantics are best
    /// explained by the elaboration:
    ///
    /// ```ignore (MIR)
    /// BB0 {
    ///   DropAndReplace(P <- V, goto BB1, unwind BB2)
    /// }
    /// ```
    ///
    /// becomes
    ///
    /// ```ignore (MIR)
    /// BB0 {
    ///   Drop(P, goto BB1, unwind BB2)
    /// }
    /// BB1 {
    ///   // P is now uninitialized
    ///   P <- V
    /// }
    /// BB2 {
    ///   // P is now uninitialized -- its dtor panicked
    ///   P <- V
    /// }
    /// ```
    ///
    /// Disallowed after drop elaboration.
    DropAndReplace {
        place: Place<'db>,
        value: Operand<'db>,
        target: BasicBlockId<'db>,
        unwind: Option<BasicBlockId<'db>>,
    },

    /// Roughly speaking, evaluates the `func` operand and the arguments, and starts execution of
    /// the referred to function. The operand types must match the argument types of the function.
    /// The return place type must match the return type. The type of the `func` operand must be
    /// callable, meaning either a function pointer, a function type, or a closure type.
    ///
    /// **Needs clarification**: The exact semantics of this. Current backends rely on `move`
    /// operands not aliasing the return place. It is unclear how this is justified in MIR, see
    /// [#71117].
    ///
    /// [#71117]: https://github.com/rust-lang/rust/issues/71117
    Call {
        /// The function that’s being called.
        func: Operand<'db>,
        /// Arguments the function is called with.
        /// These are owned by the callee, which is free to modify them.
        /// This allows the memory occupied by "by-value" arguments to be
        /// reused across function calls without duplicating the contents.
        args: Box<[Operand<'db>]>,
        /// Where the returned value will be written
        destination: Place<'db>,
        /// Where to go after this call returns. If none, the call necessarily diverges.
        target: Option<BasicBlockId<'db>>,
        /// Cleanups to be done if the call unwinds.
        cleanup: Option<BasicBlockId<'db>>,
        /// `true` if this is from a call in HIR rather than from an overloaded
        /// operator. True for overloaded function call.
        from_hir_call: bool,
        // This `Span` is the span of the function, without the dot and receiver
        // (e.g. `foo(a, b)` in `x.foo(a, b)`
        //fn_span: Span,
    },

    /// Evaluates the operand, which must have type `bool`. If it is not equal to `expected`,
    /// initiates a panic. Initiating a panic corresponds to a `Call` terminator with some
    /// unspecified constant as the function to call, all the operands stored in the `AssertMessage`
    /// as parameters, and `None` for the destination. Keep in mind that the `cleanup` path is not
    /// necessarily executed even in the case of a panic, for example in `-C panic=abort`. If the
    /// assertion does not fail, execution continues at the specified basic block.
    Assert {
        cond: Operand<'db>,
        expected: bool,
        //msg: AssertMessage,
        target: BasicBlockId<'db>,
        cleanup: Option<BasicBlockId<'db>>,
    },

    /// Marks a suspend point.
    ///
    /// Like `Return` terminators in coroutine bodies, this computes `value` and then a
    /// `CoroutineState::Yielded(value)` as if by `Aggregate` rvalue. That value is then assigned to
    /// the return place of the function calling this one, and execution continues in the calling
    /// function. When next invoked with the same first argument, execution of this function
    /// continues at the `resume` basic block, with the second argument written to the `resume_arg`
    /// place. If the coroutine is dropped before then, the `drop` basic block is invoked.
    ///
    /// Not permitted in bodies that are not coroutine bodies, or after coroutine lowering.
    ///
    /// **Needs clarification**: What about the evaluation order of the `resume_arg` and `value`?
    Yield {
        /// The value to return.
        value: Operand<'db>,
        /// Where to resume to.
        resume: BasicBlockId<'db>,
        /// The place to store the resume argument in.
        resume_arg: Place<'db>,
        /// Cleanup to be done if the coroutine is dropped at this suspend point.
        drop: Option<BasicBlockId<'db>>,
    },

    /// Indicates the end of dropping a coroutine.
    ///
    /// Semantically just a `return` (from the coroutines drop glue). Only permitted in the same situations
    /// as `yield`.
    ///
    /// **Needs clarification**: Is that even correct? The coroutine drop code is always confusing
    /// to me, because it's not even really in the current body.
    ///
    /// **Needs clarification**: Are there type system constraints on these terminators? Should
    /// there be a "block type" like `cleanup` blocks for them?
    CoroutineDrop,

    /// A block where control flow only ever takes one real path, but borrowck needs to be more
    /// conservative.
    ///
    /// At runtime this is semantically just a goto.
    ///
    /// Disallowed after drop elaboration.
    FalseEdge {
        /// The target normal control flow will take.
        real_target: BasicBlockId<'db>,
        /// A block control flow could conceptually jump to, but won't in
        /// practice.
        imaginary_target: BasicBlockId<'db>,
    },

    /// A terminator for blocks that only take one path in reality, but where we reserve the right
    /// to unwind in borrowck, even if it won't happen in practice. This can arise in infinite loops
    /// with no function calls for example.
    ///
    /// At runtime this is semantically just a goto.
    ///
    /// Disallowed after drop elaboration.
    FalseUnwind {
        /// The target normal control flow will take.
        real_target: BasicBlockId<'db>,
        /// The imaginary cleanup block link. This particular path will never be taken
        /// in practice, but in order to avoid fragility we want to always
        /// consider it in borrowck. We don't want to accept programs which
        /// pass borrowck only when `panic=abort` or some assertions are disabled
        /// due to release vs. debug mode builds. This needs to be an `Option` because
        /// of the `remove_noop_landing_pads` and `abort_unwinding_calls` passes.
        unwind: Option<BasicBlockId<'db>>,
    },
}

// Order of variants in this enum matter: they are used to compare borrow kinds.
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Shared,

    /// The immediately borrowed place must be immutable, but projections from
    /// it don't need to be. For example, a shallow borrow of `a.b` doesn't
    /// conflict with a mutable borrow of `a.b.c`.
    ///
    /// This is used when lowering matches: when matching on a place we want to
    /// ensure that place have the same value from the start of the match until
    /// an arm is selected. This prevents this code from compiling:
    /// ```compile_fail,E0510
    /// let mut x = &Some(0);
    /// match *x {
    ///     None => (),
    ///     Some(_) if { x = &None; false } => (),
    ///     Some(_) => (),
    /// }
    /// ```
    /// This can't be a shared borrow because mutably borrowing (*x as Some).0
    /// should not prevent `if let None = x { ... }`, for example, because the
    /// mutating `(*x as Some).0` can't affect the discriminant of `x`.
    /// We can also report errors with this kind of borrow differently.
    Shallow,

    /// Data is mutable and not aliasable.
    Mut { kind: MutBorrowKind },
}

// Order of variants in this enum matter: they are used to compare borrow kinds.
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum MutBorrowKind {
    /// Data must be immutable but not aliasable. This kind of borrow cannot currently
    /// be expressed by the user and is used only in implicit closure bindings.
    ClosureCapture,
    Default,
    /// This borrow arose from method-call auto-ref
    /// (i.e., adjustment::Adjust::Borrow).
    TwoPhasedBorrow,
}

impl BorrowKind {
    fn from_hir(m: hir_def::type_ref::Mutability) -> Self {
        match m {
            hir_def::type_ref::Mutability::Shared => BorrowKind::Shared,
            hir_def::type_ref::Mutability::Mut => BorrowKind::Mut { kind: MutBorrowKind::Default },
        }
    }

    fn from_rustc(m: rustc_ast_ir::Mutability) -> Self {
        match m {
            rustc_ast_ir::Mutability::Not => BorrowKind::Shared,
            rustc_ast_ir::Mutability::Mut => BorrowKind::Mut { kind: MutBorrowKind::Default },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BinOp {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    ///
    /// Division by zero is UB, because the compiler should have inserted checks
    /// prior to this.
    Div,
    /// The `%` operator (modulus)
    ///
    /// Using zero as the modulus (second operand) is UB, because the compiler
    /// should have inserted checks prior to this.
    Rem,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    ///
    /// The offset is truncated to the size of the first operand before shifting.
    Shl,
    /// The `>>` operator (shift right)
    ///
    /// The offset is truncated to the size of the first operand before shifting.
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
    /// The `ptr.offset` operator
    Offset,
}

impl BinOp {
    fn run_compare<T: PartialEq + PartialOrd>(&self, l: T, r: T) -> bool {
        match self {
            BinOp::Ge => l >= r,
            BinOp::Gt => l > r,
            BinOp::Le => l <= r,
            BinOp::Lt => l < r,
            BinOp::Eq => l == r,
            BinOp::Ne => l != r,
            x => panic!("`run_compare` called on operator {x:?}"),
        }
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Ne => "!=",
            BinOp::Ge => ">=",
            BinOp::Gt => ">",
            BinOp::Offset => "`offset`",
        })
    }
}

impl From<hir_def::hir::ArithOp> for BinOp {
    fn from(value: hir_def::hir::ArithOp) -> Self {
        match value {
            hir_def::hir::ArithOp::Add => BinOp::Add,
            hir_def::hir::ArithOp::Mul => BinOp::Mul,
            hir_def::hir::ArithOp::Sub => BinOp::Sub,
            hir_def::hir::ArithOp::Div => BinOp::Div,
            hir_def::hir::ArithOp::Rem => BinOp::Rem,
            hir_def::hir::ArithOp::Shl => BinOp::Shl,
            hir_def::hir::ArithOp::Shr => BinOp::Shr,
            hir_def::hir::ArithOp::BitXor => BinOp::BitXor,
            hir_def::hir::ArithOp::BitOr => BinOp::BitOr,
            hir_def::hir::ArithOp::BitAnd => BinOp::BitAnd,
        }
    }
}

impl From<hir_def::hir::CmpOp> for BinOp {
    fn from(value: hir_def::hir::CmpOp) -> Self {
        match value {
            hir_def::hir::CmpOp::Eq { negated: false } => BinOp::Eq,
            hir_def::hir::CmpOp::Eq { negated: true } => BinOp::Ne,
            hir_def::hir::CmpOp::Ord { ordering: Ordering::Greater, strict: false } => BinOp::Ge,
            hir_def::hir::CmpOp::Ord { ordering: Ordering::Greater, strict: true } => BinOp::Gt,
            hir_def::hir::CmpOp::Ord { ordering: Ordering::Less, strict: false } => BinOp::Le,
            hir_def::hir::CmpOp::Ord { ordering: Ordering::Less, strict: true } => BinOp::Lt,
        }
    }
}

impl<'db> From<Operand<'db>> for Rvalue<'db> {
    fn from(x: Operand<'db>) -> Self {
        Self::Use(x)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CastKind {
    /// An exposing pointer to address cast. A cast between a pointer and an integer type, or
    /// between a function pointer and an integer type.
    /// See the docs on `expose_addr` for more details.
    PointerExposeAddress,
    /// An address-to-pointer cast that picks up an exposed provenance.
    /// See the docs on `from_exposed_addr` for more details.
    PointerFromExposedAddress,
    /// All sorts of pointer-to-pointer casts. Note that reference-to-raw-ptr casts are
    /// translated into `&raw mut/const *r`, i.e., they are not actually casts.
    PtrToPtr,
    /// Pointer related casts that are done by coercions.
    PointerCoercion(PointerCast),
    /// Cast into a dyn* object.
    DynStar,
    IntToInt,
    FloatToInt,
    FloatToFloat,
    IntToFloat,
    FnPtrToPtr,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Rvalue<'db> {
    /// Yields the operand unchanged
    Use(Operand<'db>),

    /// Creates an array where each element is the value of the operand.
    ///
    /// Corresponds to source code like `[x; 32]`.
    Repeat(Operand<'db>, Const<'db>),

    /// Creates a reference of the indicated kind to the place.
    ///
    /// There is not much to document here, because besides the obvious parts the semantics of this
    /// are essentially entirely a part of the aliasing model. There are many UCG issues discussing
    /// exactly what the behavior of this operation should be.
    ///
    /// `Shallow` borrows are disallowed after drop lowering.
    Ref(BorrowKind, Place<'db>),

    /// Creates a pointer/reference to the given thread local.
    ///
    /// The yielded type is a `*mut T` if the static is mutable, otherwise if the static is extern a
    /// `*const T`, and if neither of those apply a `&T`.
    ///
    /// **Note:** This is a runtime operation that actually executes code and is in this sense more
    /// like a function call. Also, eliminating dead stores of this rvalue causes `fn main() {}` to
    /// SIGILL for some reason that I (JakobDegen) never got a chance to look into.
    ///
    /// **Needs clarification**: Are there weird additional semantics here related to the runtime
    /// nature of this operation?
    // ThreadLocalRef(DefId),
    ThreadLocalRef(std::convert::Infallible),

    /// Creates a pointer with the indicated mutability to the place.
    ///
    /// This is generated by pointer casts like `&v as *const _` or raw address of expressions like
    /// `&raw v` or `addr_of!(v)`.
    ///
    /// Like with references, the semantics of this operation are heavily dependent on the aliasing
    /// model.
    // AddressOf(Mutability, Place),
    AddressOf(std::convert::Infallible),

    /// Yields the length of the place, as a `usize`.
    ///
    /// If the type of the place is an array, this is the array length. For slices (`[T]`, not
    /// `&[T]`) this accesses the place's metadata to determine the length. This rvalue is
    /// ill-formed for places of other types.
    Len(Place<'db>),

    /// Performs essentially all of the casts that can be performed via `as`.
    ///
    /// This allows for casts from/to a variety of types.
    ///
    /// **FIXME**: Document exactly which `CastKind`s allow which types of casts. Figure out why
    /// `ArrayToPointer` and `MutToConstPointer` are special.
    Cast(CastKind, Operand<'db>, Ty<'db>),

    // FIXME link to `pointer::offset` when it hits stable.
    /// * `Offset` has the same semantics as `pointer::offset`, except that the second
    ///   parameter may be a `usize` as well.
    /// * The comparison operations accept `bool`s, `char`s, signed or unsigned integers, floats,
    ///   raw pointers, or function pointers and return a `bool`. The types of the operands must be
    ///   matching, up to the usual caveat of the lifetimes in function pointers.
    /// * Left and right shift operations accept signed or unsigned integers not necessarily of the
    ///   same type and return a value of the same type as their LHS. Like in Rust, the RHS is
    ///   truncated as needed.
    /// * The `Bit*` operations accept signed integers, unsigned integers, or bools with matching
    ///   types and return a value of that type.
    /// * The remaining operations accept signed integers, unsigned integers, or floats with
    ///   matching types and return a value of that type.
    //BinaryOp(BinOp, Box<(Operand, Operand)>),
    BinaryOp(std::convert::Infallible),

    /// Same as `BinaryOp`, but yields `(T, bool)` with a `bool` indicating an error condition.
    ///
    /// When overflow checking is disabled and we are generating run-time code, the error condition
    /// is false. Otherwise, and always during CTFE, the error condition is determined as described
    /// below.
    ///
    /// For addition, subtraction, and multiplication on integers the error condition is set when
    /// the infinite precision result would be unequal to the actual result.
    ///
    /// For shift operations on integers the error condition is set when the value of right-hand
    /// side is greater than or equal to the number of bits in the type of the left-hand side, or
    /// when the value of right-hand side is negative.
    ///
    /// Other combinations of types and operators are unsupported.
    CheckedBinaryOp(BinOp, Operand<'db>, Operand<'db>),

    /// Computes a value as described by the operation.
    //NullaryOp(NullOp, Ty),
    NullaryOp(std::convert::Infallible),

    /// Exactly like `BinaryOp`, but less operands.
    ///
    /// Also does two's-complement arithmetic. Negation requires a signed integer or a float;
    /// bitwise not requires a signed integer, unsigned integer, or bool. Both operation kinds
    /// return a value with the same type as their operand.
    UnaryOp(UnOp, Operand<'db>),

    /// Computes the discriminant of the place, returning it as an integer of type
    /// [`discriminant_ty`]. Returns zero for types without discriminant.
    ///
    /// The validity requirements for the underlying value are undecided for this rvalue, see
    /// [#91095]. Note too that the value of the discriminant is not the same thing as the
    /// variant index; use [`discriminant_for_variant`] to convert.
    ///
    /// [`discriminant_ty`]: crate::ty::Ty::discriminant_ty
    /// [#91095]: https://github.com/rust-lang/rust/issues/91095
    /// [`discriminant_for_variant`]: crate::ty::Ty::discriminant_for_variant
    Discriminant(Place<'db>),

    /// Creates an aggregate value, like a tuple or struct.
    ///
    /// This is needed because dataflow analysis needs to distinguish
    /// `dest = Foo { x: ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case that `Foo`
    /// has a destructor.
    ///
    /// Disallowed after deaggregation for all aggregate kinds except `Array` and `Coroutine`. After
    /// coroutine lowering, `Coroutine` aggregate kinds are disallowed too.
    Aggregate(AggregateKind<'db>, Box<[Operand<'db>]>),

    /// Transmutes a `*mut u8` into shallow-initialized `Box<T>`.
    ///
    /// This is different from a normal transmute because dataflow analysis will treat the box as
    /// initialized but its content as uninitialized. Like other pointer casts, this in general
    /// affects alias analysis.
    ShallowInitBox(Operand<'db>, Ty<'db>),

    /// NON STANDARD: allocates memory with the type's layout, and shallow init the box with the resulting pointer.
    ShallowInitBoxWithAlloc(Ty<'db>),

    /// A CopyForDeref is equivalent to a read from a place at the
    /// codegen level, but is treated specially by drop elaboration. When such a read happens, it
    /// is guaranteed (via nature of the mir_opt `Derefer` in rustc_mir_transform/src/deref_separator)
    /// that the only use of the returned value is a deref operation, immediately
    /// followed by one or more projections. Drop elaboration treats this rvalue as if the
    /// read never happened and just projects further. This allows simplifying various MIR
    /// optimizations and codegen backends that previously had to handle deref operations anywhere
    /// in a place.
    CopyForDeref(Place<'db>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatementKind<'db> {
    Assign(Place<'db>, Rvalue<'db>),
    FakeRead(Place<'db>),
    //SetDiscriminant {
    //    place: Box<Place>,
    //    variant_index: VariantIdx,
    //},
    Deinit(Place<'db>),
    StorageLive(LocalId<'db>),
    StorageDead(LocalId<'db>),
    //Retag(RetagKind, Box<Place>),
    //AscribeUserType(Place, UserTypeProjection, Variance),
    //Intrinsic(Box<NonDivergingIntrinsic>),
    Nop,
}
impl<'db> StatementKind<'db> {
    fn with_span(self, span: MirSpan) -> Statement<'db> {
        Statement { kind: self, span }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Statement<'db> {
    pub kind: StatementKind<'db>,
    pub span: MirSpan,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BasicBlock<'db> {
    /// List of statements in this block.
    pub statements: Vec<Statement<'db>>,

    /// Terminator for this block.
    ///
    /// N.B., this should generally ONLY be `None` during construction.
    /// Therefore, you should generally access it via the
    /// `terminator()` or `terminator_mut()` methods. The only
    /// exception is that certain passes, such as `simplify_cfg`, swap
    /// out the terminator temporarily with `None` while they continue
    /// to recurse over the set of basic blocks.
    pub terminator: Option<Terminator<'db>>,

    /// If true, this block lies on an unwind path. This is used
    /// during codegen where distinct kinds of basic blocks may be
    /// generated (particularly for MSVC cleanup). Unwind blocks must
    /// only branch to other unwind blocks.
    pub is_cleanup: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirBody<'db> {
    pub projection_store: ProjectionStore<'db>,
    pub basic_blocks: Arena<BasicBlock<'db>>,
    pub locals: Arena<Local<'db>>,
    pub start_block: BasicBlockId<'db>,
    pub owner: DefWithBodyId,
    pub binding_locals: ArenaMap<BindingId, LocalId<'db>>,
    pub param_locals: Vec<LocalId<'db>>,
    /// This field stores the closures directly owned by this body. It is used
    /// in traversing every mir body.
    pub closures: Vec<InternedClosureId>,
}

impl<'db> MirBody<'db> {
    pub fn local_to_binding_map(&self) -> ArenaMap<LocalId<'db>, BindingId> {
        self.binding_locals.iter().map(|(it, y)| (*y, it)).collect()
    }

    fn walk_places(&mut self, mut f: impl FnMut(&mut Place<'db>, &mut ProjectionStore<'db>)) {
        fn for_operand<'db>(
            op: &mut Operand<'db>,
            f: &mut impl FnMut(&mut Place<'db>, &mut ProjectionStore<'db>),
            store: &mut ProjectionStore<'db>,
        ) {
            match &mut op.kind {
                OperandKind::Copy(p) | OperandKind::Move(p) => {
                    f(p, store);
                }
                OperandKind::Constant { .. } | OperandKind::Static(_) => (),
            }
        }
        for (_, block) in self.basic_blocks.iter_mut() {
            for statement in &mut block.statements {
                match &mut statement.kind {
                    StatementKind::Assign(p, r) => {
                        f(p, &mut self.projection_store);
                        match r {
                            Rvalue::ShallowInitBoxWithAlloc(_) => (),
                            Rvalue::ShallowInitBox(o, _)
                            | Rvalue::UnaryOp(_, o)
                            | Rvalue::Cast(_, o, _)
                            | Rvalue::Repeat(o, _)
                            | Rvalue::Use(o) => for_operand(o, &mut f, &mut self.projection_store),
                            Rvalue::CopyForDeref(p)
                            | Rvalue::Discriminant(p)
                            | Rvalue::Len(p)
                            | Rvalue::Ref(_, p) => f(p, &mut self.projection_store),
                            Rvalue::CheckedBinaryOp(_, o1, o2) => {
                                for_operand(o1, &mut f, &mut self.projection_store);
                                for_operand(o2, &mut f, &mut self.projection_store);
                            }
                            Rvalue::Aggregate(_, ops) => {
                                for op in ops.iter_mut() {
                                    for_operand(op, &mut f, &mut self.projection_store);
                                }
                            }
                            Rvalue::ThreadLocalRef(n)
                            | Rvalue::AddressOf(n)
                            | Rvalue::BinaryOp(n)
                            | Rvalue::NullaryOp(n) => match *n {},
                        }
                    }
                    StatementKind::FakeRead(p) | StatementKind::Deinit(p) => {
                        f(p, &mut self.projection_store)
                    }
                    StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::Nop => (),
                }
            }
            match &mut block.terminator {
                Some(x) => match &mut x.kind {
                    TerminatorKind::SwitchInt { discr, .. } => {
                        for_operand(discr, &mut f, &mut self.projection_store)
                    }
                    TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. }
                    | TerminatorKind::Goto { .. }
                    | TerminatorKind::UnwindResume
                    | TerminatorKind::CoroutineDrop
                    | TerminatorKind::Abort
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable => (),
                    TerminatorKind::Drop { place, .. } => {
                        f(place, &mut self.projection_store);
                    }
                    TerminatorKind::DropAndReplace { place, value, .. } => {
                        f(place, &mut self.projection_store);
                        for_operand(value, &mut f, &mut self.projection_store);
                    }
                    TerminatorKind::Call { func, args, destination, .. } => {
                        for_operand(func, &mut f, &mut self.projection_store);
                        args.iter_mut()
                            .for_each(|x| for_operand(x, &mut f, &mut self.projection_store));
                        f(destination, &mut self.projection_store);
                    }
                    TerminatorKind::Assert { cond, .. } => {
                        for_operand(cond, &mut f, &mut self.projection_store);
                    }
                    TerminatorKind::Yield { value, resume_arg, .. } => {
                        for_operand(value, &mut f, &mut self.projection_store);
                        f(resume_arg, &mut self.projection_store);
                    }
                },
                None => (),
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        let MirBody {
            basic_blocks,
            locals,
            start_block: _,
            owner: _,
            binding_locals,
            param_locals,
            closures,
            projection_store,
        } = self;
        projection_store.shrink_to_fit();
        basic_blocks.shrink_to_fit();
        locals.shrink_to_fit();
        binding_locals.shrink_to_fit();
        param_locals.shrink_to_fit();
        closures.shrink_to_fit();
        for (_, b) in basic_blocks.iter_mut() {
            let BasicBlock { statements, terminator: _, is_cleanup: _ } = b;
            statements.shrink_to_fit();
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MirSpan {
    ExprId(ExprId),
    PatId(PatId),
    BindingId(BindingId),
    SelfParam,
    Unknown,
}

impl MirSpan {
    pub fn is_ref_span(&self, body: &Body) -> bool {
        match *self {
            MirSpan::ExprId(expr) => matches!(body[expr], Expr::Ref { .. }),
            // FIXME: Figure out if this is correct wrt. match ergonomics.
            MirSpan::BindingId(binding) => {
                matches!(body[binding].mode, BindingAnnotation::Ref | BindingAnnotation::RefMut)
            }
            MirSpan::PatId(_) | MirSpan::SelfParam | MirSpan::Unknown => false,
        }
    }
}

impl_from!(ExprId, PatId for MirSpan);

impl From<&ExprId> for MirSpan {
    fn from(value: &ExprId) -> Self {
        (*value).into()
    }
}
