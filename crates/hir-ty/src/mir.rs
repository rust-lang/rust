//! MIR definitions and implementation

use std::{fmt::Display, iter};

use crate::{
    consteval::usize_const,
    db::HirDatabase,
    display::HirDisplay,
    infer::{normalize, PointerCast},
    lang_items::is_box,
    mapping::ToChalk,
    CallableDefId, ClosureId, Const, ConstScalar, InferenceResult, Interner, MemoryMap,
    Substitution, TraitEnvironment, Ty, TyKind,
};
use base_db::CrateId;
use chalk_ir::Mutability;
use hir_def::{
    hir::{BindingId, Expr, ExprId, Ordering, PatId},
    DefWithBodyId, FieldId, StaticId, UnionId, VariantId,
};
use la_arena::{Arena, ArenaMap, Idx, RawIdx};

mod eval;
mod lower;
mod borrowck;
mod pretty;
mod monomorphization;

pub use borrowck::{borrowck_query, BorrowckResult, MutabilityReason};
pub use eval::{
    interpret_mir, pad16, render_const_using_debug_impl, Evaluator, MirEvalError, VTableMap,
};
pub use lower::{
    lower_to_mir, mir_body_for_closure_query, mir_body_query, mir_body_recover, MirLowerError,
};
pub use monomorphization::{
    monomorphize_mir_body_bad, monomorphized_mir_body_for_closure_query,
    monomorphized_mir_body_query, monomorphized_mir_body_recover,
};
use smallvec::{smallvec, SmallVec};
use stdx::{impl_from, never};
use triomphe::Arc;

use super::consteval::{intern_const_scalar, try_const_usize};

pub type BasicBlockId = Idx<BasicBlock>;
pub type LocalId = Idx<Local>;

fn return_slot() -> LocalId {
    LocalId::from_raw(RawIdx::from(0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub ty: Ty,
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
pub enum Operand {
    /// Creates a value by loading the given place.
    ///
    /// Before drop elaboration, the type of the place must be `Copy`. After drop elaboration there
    /// is no such requirement.
    Copy(Place),

    /// Creates a value by performing loading the place, just like the `Copy` operand.
    ///
    /// This *may* additionally overwrite the place with `uninit` bytes, depending on how we decide
    /// in [UCG#188]. You should not emit MIR that may attempt a subsequent second load of this
    /// place without first re-initializing it.
    ///
    /// [UCG#188]: https://github.com/rust-lang/unsafe-code-guidelines/issues/188
    Move(Place),
    /// Constants are already semantically values, and remain unchanged.
    Constant(Const),
    /// NON STANDARD: This kind of operand returns an immutable reference to that static memory. Rustc
    /// handles it with the `Constant` variant somehow.
    Static(StaticId),
}

impl Operand {
    fn from_concrete_const(data: Vec<u8>, memory_map: MemoryMap, ty: Ty) -> Self {
        Operand::Constant(intern_const_scalar(ConstScalar::Bytes(data, memory_map), ty))
    }

    fn from_bytes(data: Vec<u8>, ty: Ty) -> Self {
        Operand::from_concrete_const(data, MemoryMap::default(), ty)
    }

    fn const_zst(ty: Ty) -> Operand {
        Self::from_bytes(vec![], ty)
    }

    fn from_fn(
        db: &dyn HirDatabase,
        func_id: hir_def::FunctionId,
        generic_args: Substitution,
    ) -> Operand {
        let ty =
            chalk_ir::TyKind::FnDef(CallableDefId::FunctionId(func_id).to_chalk(db), generic_args)
                .intern(Interner);
        Operand::from_bytes(vec![], ty)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProjectionElem<V, T> {
    Deref,
    Field(FieldId),
    // FIXME: get rid of this, and use FieldId for tuples and closures
    TupleOrClosureField(usize),
    Index(V),
    ConstantIndex { offset: u64, from_end: bool },
    Subslice { from: u64, to: u64 },
    //Downcast(Option<Symbol>, VariantIdx),
    OpaqueCast(T),
}

impl<V, T> ProjectionElem<V, T> {
    pub fn projected_ty(
        &self,
        mut base: Ty,
        db: &dyn HirDatabase,
        closure_field: impl FnOnce(ClosureId, &Substitution, usize) -> Ty,
        krate: CrateId,
    ) -> Ty {
        if matches!(base.data(Interner).kind, TyKind::Alias(_) | TyKind::AssociatedType(..)) {
            base = normalize(
                db,
                // FIXME: we should get this from caller
                Arc::new(TraitEnvironment::empty(krate)),
                base,
            );
        }
        match self {
            ProjectionElem::Deref => match &base.data(Interner).kind {
                TyKind::Raw(_, inner) | TyKind::Ref(_, _, inner) => inner.clone(),
                TyKind::Adt(adt, subst) if is_box(db, adt.0) => {
                    subst.at(Interner, 0).assert_ty_ref(Interner).clone()
                }
                _ => {
                    never!("Overloaded deref on type {} is not a projection", base.display(db));
                    return TyKind::Error.intern(Interner);
                }
            },
            ProjectionElem::Field(f) => match &base.data(Interner).kind {
                TyKind::Adt(_, subst) => {
                    db.field_types(f.parent)[f.local_id].clone().substitute(Interner, subst)
                }
                _ => {
                    never!("Only adt has field");
                    return TyKind::Error.intern(Interner);
                }
            },
            ProjectionElem::TupleOrClosureField(f) => match &base.data(Interner).kind {
                TyKind::Tuple(_, subst) => subst
                    .as_slice(Interner)
                    .get(*f)
                    .map(|x| x.assert_ty_ref(Interner))
                    .cloned()
                    .unwrap_or_else(|| {
                        never!("Out of bound tuple field");
                        TyKind::Error.intern(Interner)
                    }),
                TyKind::Closure(id, subst) => closure_field(*id, subst, *f),
                _ => {
                    never!("Only tuple or closure has tuple or closure field");
                    return TyKind::Error.intern(Interner);
                }
            },
            ProjectionElem::ConstantIndex { .. } | ProjectionElem::Index(_) => {
                match &base.data(Interner).kind {
                    TyKind::Array(inner, _) | TyKind::Slice(inner) => inner.clone(),
                    _ => {
                        never!("Overloaded index is not a projection");
                        return TyKind::Error.intern(Interner);
                    }
                }
            }
            &ProjectionElem::Subslice { from, to } => match &base.data(Interner).kind {
                TyKind::Array(inner, c) => {
                    let next_c = usize_const(
                        db,
                        match try_const_usize(db, c) {
                            None => None,
                            Some(x) => x.checked_sub(u128::from(from + to)),
                        },
                        krate,
                    );
                    TyKind::Array(inner.clone(), next_c).intern(Interner)
                }
                TyKind::Slice(_) => base.clone(),
                _ => {
                    never!("Subslice projection should only happen on slice and array");
                    return TyKind::Error.intern(Interner);
                }
            },
            ProjectionElem::OpaqueCast(_) => {
                never!("We don't emit these yet");
                return TyKind::Error.intern(Interner);
            }
        }
    }
}

type PlaceElem = ProjectionElem<LocalId, Ty>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    pub local: LocalId,
    pub projection: Box<[PlaceElem]>,
}

impl Place {
    fn is_parent(&self, child: &Place) -> bool {
        self.local == child.local && child.projection.starts_with(&self.projection)
    }

    fn iterate_over_parents(&self) -> impl Iterator<Item = Place> + '_ {
        (0..self.projection.len())
            .map(|x| &self.projection[0..x])
            .map(|x| Place { local: self.local, projection: x.to_vec().into() })
    }

    fn project(&self, projection: PlaceElem) -> Place {
        Place {
            local: self.local,
            projection: self.projection.iter().cloned().chain([projection]).collect(),
        }
    }
}

impl From<LocalId> for Place {
    fn from(local: LocalId) -> Self {
        Self { local, projection: vec![].into() }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AggregateKind {
    /// The type is of the element
    Array(Ty),
    /// The type is of the tuple
    Tuple(Ty),
    Adt(VariantId, Substitution),
    Union(UnionId, FieldId),
    Closure(Ty),
    //Generator(LocalDefId, SubstsRef, Movability),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SwitchTargets {
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
    targets: SmallVec<[BasicBlockId; 2]>,
}

impl SwitchTargets {
    /// Creates switch targets from an iterator of values and target blocks.
    ///
    /// The iterator may be empty, in which case the `SwitchInt` instruction is equivalent to
    /// `goto otherwise;`.
    pub fn new(
        targets: impl Iterator<Item = (u128, BasicBlockId)>,
        otherwise: BasicBlockId,
    ) -> Self {
        let (values, mut targets): (SmallVec<_>, SmallVec<_>) = targets.unzip();
        targets.push(otherwise);
        Self { values, targets }
    }

    /// Builds a switch targets definition that jumps to `then` if the tested value equals `value`,
    /// and to `else_` if not.
    pub fn static_if(value: u128, then: BasicBlockId, else_: BasicBlockId) -> Self {
        Self { values: smallvec![value], targets: smallvec![then, else_] }
    }

    /// Returns the fallback target that is jumped to when none of the values match the operand.
    pub fn otherwise(&self) -> BasicBlockId {
        *self.targets.last().unwrap()
    }

    /// Returns an iterator over the switch targets.
    ///
    /// The iterator will yield tuples containing the value and corresponding target to jump to, not
    /// including the `otherwise` fallback target.
    ///
    /// Note that this may yield 0 elements. Only the `otherwise` branch is mandatory.
    pub fn iter(&self) -> impl Iterator<Item = (u128, BasicBlockId)> + '_ {
        iter::zip(&self.values, &self.targets).map(|(x, y)| (*x, *y))
    }

    /// Returns a slice with all possible jump targets (including the fallback target).
    pub fn all_targets(&self) -> &[BasicBlockId] {
        &self.targets
    }

    /// Finds the `BasicBlock` to which this `SwitchInt` will branch given the
    /// specific value. This cannot fail, as it'll return the `otherwise`
    /// branch if there's not a specific match for the value.
    pub fn target_for_value(&self, value: u128) -> BasicBlockId {
        self.iter().find_map(|(v, t)| (v == value).then_some(t)).unwrap_or_else(|| self.otherwise())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Terminator {
    pub span: MirSpan,
    pub kind: TerminatorKind,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TerminatorKind {
    /// Block has one successor; we continue execution there.
    Goto { target: BasicBlockId },

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
        discr: Operand,

        targets: SwitchTargets,
    },

    /// Indicates that the landing pad is finished and that the process should continue unwinding.
    ///
    /// Like a return, this marks the end of this invocation of the function.
    ///
    /// Only permitted in cleanup blocks. `Resume` is not permitted with `-C unwind=abort` after
    /// deaggregation runs.
    Resume,

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
    /// If the body is a generator body, this has slightly different semantics; it instead causes a
    /// `GeneratorState::Returned(_0)` to be created (as if by an `Aggregate` rvalue) and assigned
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
    Drop { place: Place, target: BasicBlockId, unwind: Option<BasicBlockId> },

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
        place: Place,
        value: Operand,
        target: BasicBlockId,
        unwind: Option<BasicBlockId>,
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
        func: Operand,
        /// Arguments the function is called with.
        /// These are owned by the callee, which is free to modify them.
        /// This allows the memory occupied by "by-value" arguments to be
        /// reused across function calls without duplicating the contents.
        args: Box<[Operand]>,
        /// Where the returned value will be written
        destination: Place,
        /// Where to go after this call returns. If none, the call necessarily diverges.
        target: Option<BasicBlockId>,
        /// Cleanups to be done if the call unwinds.
        cleanup: Option<BasicBlockId>,
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
        cond: Operand,
        expected: bool,
        //msg: AssertMessage,
        target: BasicBlockId,
        cleanup: Option<BasicBlockId>,
    },

    /// Marks a suspend point.
    ///
    /// Like `Return` terminators in generator bodies, this computes `value` and then a
    /// `GeneratorState::Yielded(value)` as if by `Aggregate` rvalue. That value is then assigned to
    /// the return place of the function calling this one, and execution continues in the calling
    /// function. When next invoked with the same first argument, execution of this function
    /// continues at the `resume` basic block, with the second argument written to the `resume_arg`
    /// place. If the generator is dropped before then, the `drop` basic block is invoked.
    ///
    /// Not permitted in bodies that are not generator bodies, or after generator lowering.
    ///
    /// **Needs clarification**: What about the evaluation order of the `resume_arg` and `value`?
    Yield {
        /// The value to return.
        value: Operand,
        /// Where to resume to.
        resume: BasicBlockId,
        /// The place to store the resume argument in.
        resume_arg: Place,
        /// Cleanup to be done if the generator is dropped at this suspend point.
        drop: Option<BasicBlockId>,
    },

    /// Indicates the end of dropping a generator.
    ///
    /// Semantically just a `return` (from the generators drop glue). Only permitted in the same situations
    /// as `yield`.
    ///
    /// **Needs clarification**: Is that even correct? The generator drop code is always confusing
    /// to me, because it's not even really in the current body.
    ///
    /// **Needs clarification**: Are there type system constraints on these terminators? Should
    /// there be a "block type" like `cleanup` blocks for them?
    GeneratorDrop,

    /// A block where control flow only ever takes one real path, but borrowck needs to be more
    /// conservative.
    ///
    /// At runtime this is semantically just a goto.
    ///
    /// Disallowed after drop elaboration.
    FalseEdge {
        /// The target normal control flow will take.
        real_target: BasicBlockId,
        /// A block control flow could conceptually jump to, but won't in
        /// practice.
        imaginary_target: BasicBlockId,
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
        real_target: BasicBlockId,
        /// The imaginary cleanup block link. This particular path will never be taken
        /// in practice, but in order to avoid fragility we want to always
        /// consider it in borrowck. We don't want to accept programs which
        /// pass borrowck only when `panic=abort` or some assertions are disabled
        /// due to release vs. debug mode builds. This needs to be an `Option` because
        /// of the `remove_noop_landing_pads` and `abort_unwinding_calls` passes.
        unwind: Option<BasicBlockId>,
    },
}

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

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure is
    /// borrowing or mutating a mutable referent, e.g.:
    /// ```
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = || *x += 5;
    /// ```
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    /// ```compile_fail,E0594
    /// struct Env<'a> { x: &'a &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &x }, fn_ptr);  // Closure is pair of env and fn
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    /// This is then illegal because you cannot mutate an `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    /// ```compile_fail,E0596
    /// struct Env<'a> { x: &'a mut &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &mut x }, fn_ptr); // changed from &x to &mut x
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    Unique,

    /// Data is mutable and not aliasable.
    Mut {
        /// `true` if this borrow arose from method-call auto-ref
        /// (i.e., `adjustment::Adjust::Borrow`).
        allow_two_phase_borrow: bool,
    },
}

impl BorrowKind {
    fn from_hir(m: hir_def::type_ref::Mutability) -> Self {
        match m {
            hir_def::type_ref::Mutability::Shared => BorrowKind::Shared,
            hir_def::type_ref::Mutability::Mut => BorrowKind::Mut { allow_two_phase_borrow: false },
        }
    }

    fn from_chalk(m: Mutability) -> Self {
        match m {
            Mutability::Not => BorrowKind::Shared,
            Mutability::Mut => BorrowKind::Mut { allow_two_phase_borrow: false },
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

impl From<Operand> for Rvalue {
    fn from(x: Operand) -> Self {
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
    Pointer(PointerCast),
    /// Cast into a dyn* object.
    DynStar,
    IntToInt,
    FloatToInt,
    FloatToFloat,
    IntToFloat,
    FnPtrToPtr,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Rvalue {
    /// Yields the operand unchanged
    Use(Operand),

    /// Creates an array where each element is the value of the operand.
    ///
    /// Corresponds to source code like `[x; 32]`.
    Repeat(Operand, Const),

    /// Creates a reference of the indicated kind to the place.
    ///
    /// There is not much to document here, because besides the obvious parts the semantics of this
    /// are essentially entirely a part of the aliasing model. There are many UCG issues discussing
    /// exactly what the behavior of this operation should be.
    ///
    /// `Shallow` borrows are disallowed after drop lowering.
    Ref(BorrowKind, Place),

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
    //ThreadLocalRef(DefId),

    /// Creates a pointer with the indicated mutability to the place.
    ///
    /// This is generated by pointer casts like `&v as *const _` or raw address of expressions like
    /// `&raw v` or `addr_of!(v)`.
    ///
    /// Like with references, the semantics of this operation are heavily dependent on the aliasing
    /// model.
    //AddressOf(Mutability, Place),

    /// Yields the length of the place, as a `usize`.
    ///
    /// If the type of the place is an array, this is the array length. For slices (`[T]`, not
    /// `&[T]`) this accesses the place's metadata to determine the length. This rvalue is
    /// ill-formed for places of other types.
    Len(Place),

    /// Performs essentially all of the casts that can be performed via `as`.
    ///
    /// This allows for casts from/to a variety of types.
    ///
    /// **FIXME**: Document exactly which `CastKind`s allow which types of casts. Figure out why
    /// `ArrayToPointer` and `MutToConstPointer` are special.
    Cast(CastKind, Operand, Ty),

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
    CheckedBinaryOp(BinOp, Operand, Operand),

    /// Computes a value as described by the operation.
    //NullaryOp(NullOp, Ty),

    /// Exactly like `BinaryOp`, but less operands.
    ///
    /// Also does two's-complement arithmetic. Negation requires a signed integer or a float;
    /// bitwise not requires a signed integer, unsigned integer, or bool. Both operation kinds
    /// return a value with the same type as their operand.
    UnaryOp(UnOp, Operand),

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
    Discriminant(Place),

    /// Creates an aggregate value, like a tuple or struct.
    ///
    /// This is needed because dataflow analysis needs to distinguish
    /// `dest = Foo { x: ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case that `Foo`
    /// has a destructor.
    ///
    /// Disallowed after deaggregation for all aggregate kinds except `Array` and `Generator`. After
    /// generator lowering, `Generator` aggregate kinds are disallowed too.
    Aggregate(AggregateKind, Box<[Operand]>),

    /// Transmutes a `*mut u8` into shallow-initialized `Box<T>`.
    ///
    /// This is different from a normal transmute because dataflow analysis will treat the box as
    /// initialized but its content as uninitialized. Like other pointer casts, this in general
    /// affects alias analysis.
    ShallowInitBox(Operand, Ty),

    /// NON STANDARD: allocates memory with the type's layout, and shallow init the box with the resulting pointer.
    ShallowInitBoxWithAlloc(Ty),

    /// A CopyForDeref is equivalent to a read from a place at the
    /// codegen level, but is treated specially by drop elaboration. When such a read happens, it
    /// is guaranteed (via nature of the mir_opt `Derefer` in rustc_mir_transform/src/deref_separator)
    /// that the only use of the returned value is a deref operation, immediately
    /// followed by one or more projections. Drop elaboration treats this rvalue as if the
    /// read never happened and just projects further. This allows simplifying various MIR
    /// optimizations and codegen backends that previously had to handle deref operations anywhere
    /// in a place.
    CopyForDeref(Place),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatementKind {
    Assign(Place, Rvalue),
    //FakeRead(Box<(FakeReadCause, Place)>),
    //SetDiscriminant {
    //    place: Box<Place>,
    //    variant_index: VariantIdx,
    //},
    Deinit(Place),
    StorageLive(LocalId),
    StorageDead(LocalId),
    //Retag(RetagKind, Box<Place>),
    //AscribeUserType(Place, UserTypeProjection, Variance),
    //Intrinsic(Box<NonDivergingIntrinsic>),
    Nop,
}
impl StatementKind {
    fn with_span(self, span: MirSpan) -> Statement {
        Statement { kind: self, span }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: MirSpan,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    /// List of statements in this block.
    pub statements: Vec<Statement>,

    /// Terminator for this block.
    ///
    /// N.B., this should generally ONLY be `None` during construction.
    /// Therefore, you should generally access it via the
    /// `terminator()` or `terminator_mut()` methods. The only
    /// exception is that certain passes, such as `simplify_cfg`, swap
    /// out the terminator temporarily with `None` while they continue
    /// to recurse over the set of basic blocks.
    pub terminator: Option<Terminator>,

    /// If true, this block lies on an unwind path. This is used
    /// during codegen where distinct kinds of basic blocks may be
    /// generated (particularly for MSVC cleanup). Unwind blocks must
    /// only branch to other unwind blocks.
    pub is_cleanup: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirBody {
    pub basic_blocks: Arena<BasicBlock>,
    pub locals: Arena<Local>,
    pub start_block: BasicBlockId,
    pub owner: DefWithBodyId,
    pub binding_locals: ArenaMap<BindingId, LocalId>,
    pub param_locals: Vec<LocalId>,
    /// This field stores the closures directly owned by this body. It is used
    /// in traversing every mir body.
    pub closures: Vec<ClosureId>,
}

impl MirBody {
    fn walk_places(&mut self, mut f: impl FnMut(&mut Place)) {
        fn for_operand(op: &mut Operand, f: &mut impl FnMut(&mut Place)) {
            match op {
                Operand::Copy(p) | Operand::Move(p) => {
                    f(p);
                }
                Operand::Constant(_) | Operand::Static(_) => (),
            }
        }
        for (_, block) in self.basic_blocks.iter_mut() {
            for statement in &mut block.statements {
                match &mut statement.kind {
                    StatementKind::Assign(p, r) => {
                        f(p);
                        match r {
                            Rvalue::ShallowInitBoxWithAlloc(_) => (),
                            Rvalue::ShallowInitBox(o, _)
                            | Rvalue::UnaryOp(_, o)
                            | Rvalue::Cast(_, o, _)
                            | Rvalue::Repeat(o, _)
                            | Rvalue::Use(o) => for_operand(o, &mut f),
                            Rvalue::CopyForDeref(p)
                            | Rvalue::Discriminant(p)
                            | Rvalue::Len(p)
                            | Rvalue::Ref(_, p) => f(p),
                            Rvalue::CheckedBinaryOp(_, o1, o2) => {
                                for_operand(o1, &mut f);
                                for_operand(o2, &mut f);
                            }
                            Rvalue::Aggregate(_, ops) => {
                                for op in ops.iter_mut() {
                                    for_operand(op, &mut f);
                                }
                            }
                        }
                    }
                    StatementKind::Deinit(p) => f(p),
                    StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::Nop => (),
                }
            }
            match &mut block.terminator {
                Some(x) => match &mut x.kind {
                    TerminatorKind::SwitchInt { discr, .. } => for_operand(discr, &mut f),
                    TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. }
                    | TerminatorKind::Goto { .. }
                    | TerminatorKind::Resume
                    | TerminatorKind::GeneratorDrop
                    | TerminatorKind::Abort
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable => (),
                    TerminatorKind::Drop { place, .. } => {
                        f(place);
                    }
                    TerminatorKind::DropAndReplace { place, value, .. } => {
                        f(place);
                        for_operand(value, &mut f);
                    }
                    TerminatorKind::Call { func, args, destination, .. } => {
                        for_operand(func, &mut f);
                        args.iter_mut().for_each(|x| for_operand(x, &mut f));
                        f(destination);
                    }
                    TerminatorKind::Assert { cond, .. } => {
                        for_operand(cond, &mut f);
                    }
                    TerminatorKind::Yield { value, resume_arg, .. } => {
                        for_operand(value, &mut f);
                        f(resume_arg);
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
        } = self;
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
    Unknown,
}

impl_from!(ExprId, PatId for MirSpan);
