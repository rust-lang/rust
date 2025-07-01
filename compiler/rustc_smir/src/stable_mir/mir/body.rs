use std::io;

use serde::Serialize;
use stable_mir::compiler_interface::with;
use stable_mir::mir::pretty::function_body;
use stable_mir::ty::{
    AdtDef, ClosureDef, CoroutineClosureDef, CoroutineDef, GenericArgs, MirConst, Movability,
    Region, RigidTy, Ty, TyConst, TyKind, VariantIdx,
};
use stable_mir::{Error, Opaque, Span, Symbol};

use crate::stable_mir;

/// The SMIR representation of a single function.
#[derive(Clone, Debug, Serialize)]
pub struct Body {
    pub blocks: Vec<BasicBlock>,

    /// Declarations of locals within the function.
    ///
    /// The first local is the return value pointer, followed by `arg_count`
    /// locals for the function arguments, followed by any user-declared
    /// variables and temporaries.
    pub(super) locals: LocalDecls,

    /// The number of arguments this function takes.
    pub(super) arg_count: usize,

    /// Debug information pertaining to user variables, including captures.
    pub var_debug_info: Vec<VarDebugInfo>,

    /// Mark an argument (which must be a tuple) as getting passed as its individual components.
    ///
    /// This is used for the "rust-call" ABI such as closures.
    pub(super) spread_arg: Option<Local>,

    /// The span that covers the entire function body.
    pub span: Span,
}

pub type BasicBlockIdx = usize;

impl Body {
    /// Constructs a `Body`.
    ///
    /// A constructor is required to build a `Body` from outside the crate
    /// because the `arg_count` and `locals` fields are private.
    pub fn new(
        blocks: Vec<BasicBlock>,
        locals: LocalDecls,
        arg_count: usize,
        var_debug_info: Vec<VarDebugInfo>,
        spread_arg: Option<Local>,
        span: Span,
    ) -> Self {
        // If locals doesn't contain enough entries, it can lead to panics in
        // `ret_local`, `arg_locals`, and `inner_locals`.
        assert!(
            locals.len() > arg_count,
            "A Body must contain at least a local for the return value and each of the function's arguments"
        );
        Self { blocks, locals, arg_count, var_debug_info, spread_arg, span }
    }

    /// Return local that holds this function's return value.
    pub fn ret_local(&self) -> &LocalDecl {
        &self.locals[RETURN_LOCAL]
    }

    /// Locals in `self` that correspond to this function's arguments.
    pub fn arg_locals(&self) -> &[LocalDecl] {
        &self.locals[1..][..self.arg_count]
    }

    /// Inner locals for this function. These are the locals that are
    /// neither the return local nor the argument locals.
    pub fn inner_locals(&self) -> &[LocalDecl] {
        &self.locals[self.arg_count + 1..]
    }

    /// Returns a mutable reference to the local that holds this function's return value.
    pub(crate) fn ret_local_mut(&mut self) -> &mut LocalDecl {
        &mut self.locals[RETURN_LOCAL]
    }

    /// Returns a mutable slice of locals corresponding to this function's arguments.
    pub(crate) fn arg_locals_mut(&mut self) -> &mut [LocalDecl] {
        &mut self.locals[1..][..self.arg_count]
    }

    /// Returns a mutable slice of inner locals for this function.
    /// Inner locals are those that are neither the return local nor the argument locals.
    pub(crate) fn inner_locals_mut(&mut self) -> &mut [LocalDecl] {
        &mut self.locals[self.arg_count + 1..]
    }

    /// Convenience function to get all the locals in this function.
    ///
    /// Locals are typically accessed via the more specific methods `ret_local`,
    /// `arg_locals`, and `inner_locals`.
    pub fn locals(&self) -> &[LocalDecl] {
        &self.locals
    }

    /// Get the local declaration for this local.
    pub fn local_decl(&self, local: Local) -> Option<&LocalDecl> {
        self.locals.get(local)
    }

    /// Get an iterator for all local declarations.
    pub fn local_decls(&self) -> impl Iterator<Item = (Local, &LocalDecl)> {
        self.locals.iter().enumerate()
    }

    /// Emit the body using the provided name for the signature.
    pub fn dump<W: io::Write>(&self, w: &mut W, fn_name: &str) -> io::Result<()> {
        function_body(w, self, fn_name)
    }

    pub fn spread_arg(&self) -> Option<Local> {
        self.spread_arg
    }
}

type LocalDecls = Vec<LocalDecl>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LocalDecl {
    pub ty: Ty,
    pub span: Span,
    pub mutability: Mutability,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Terminator {
    pub kind: TerminatorKind,
    pub span: Span,
}

impl Terminator {
    pub fn successors(&self) -> Successors {
        self.kind.successors()
    }
}

pub type Successors = Vec<BasicBlockIdx>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum TerminatorKind {
    Goto {
        target: BasicBlockIdx,
    },
    SwitchInt {
        discr: Operand,
        targets: SwitchTargets,
    },
    Resume,
    Abort,
    Return,
    Unreachable,
    Drop {
        place: Place,
        target: BasicBlockIdx,
        unwind: UnwindAction,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<BasicBlockIdx>,
        unwind: UnwindAction,
    },
    Assert {
        cond: Operand,
        expected: bool,
        msg: AssertMessage,
        target: BasicBlockIdx,
        unwind: UnwindAction,
    },
    InlineAsm {
        template: String,
        operands: Vec<InlineAsmOperand>,
        options: String,
        line_spans: String,
        destination: Option<BasicBlockIdx>,
        unwind: UnwindAction,
    },
}

impl TerminatorKind {
    pub fn successors(&self) -> Successors {
        use self::TerminatorKind::*;
        match *self {
            Call { target: Some(t), unwind: UnwindAction::Cleanup(u), .. }
            | Drop { target: t, unwind: UnwindAction::Cleanup(u), .. }
            | Assert { target: t, unwind: UnwindAction::Cleanup(u), .. }
            | InlineAsm { destination: Some(t), unwind: UnwindAction::Cleanup(u), .. } => {
                vec![t, u]
            }
            Goto { target: t }
            | Call { target: None, unwind: UnwindAction::Cleanup(t), .. }
            | Call { target: Some(t), unwind: _, .. }
            | Drop { target: t, unwind: _, .. }
            | Assert { target: t, unwind: _, .. }
            | InlineAsm { destination: None, unwind: UnwindAction::Cleanup(t), .. }
            | InlineAsm { destination: Some(t), unwind: _, .. } => {
                vec![t]
            }

            Return
            | Resume
            | Abort
            | Unreachable
            | Call { target: None, unwind: _, .. }
            | InlineAsm { destination: None, unwind: _, .. } => {
                vec![]
            }
            SwitchInt { ref targets, .. } => targets.all_targets(),
        }
    }

    pub fn unwind(&self) -> Option<&UnwindAction> {
        match *self {
            TerminatorKind::Goto { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::SwitchInt { .. } => None,
            TerminatorKind::Call { ref unwind, .. }
            | TerminatorKind::Assert { ref unwind, .. }
            | TerminatorKind::Drop { ref unwind, .. }
            | TerminatorKind::InlineAsm { ref unwind, .. } => Some(unwind),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InlineAsmOperand {
    pub in_value: Option<Operand>,
    pub out_place: Option<Place>,
    // This field has a raw debug representation of MIR's InlineAsmOperand.
    // For now we care about place/operand + the rest in a debug format.
    pub raw_rpr: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum UnwindAction {
    Continue,
    Unreachable,
    Terminate,
    Cleanup(BasicBlockIdx),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum AssertMessage {
    BoundsCheck { len: Operand, index: Operand },
    Overflow(BinOp, Operand, Operand),
    OverflowNeg(Operand),
    DivisionByZero(Operand),
    RemainderByZero(Operand),
    ResumedAfterReturn(CoroutineKind),
    ResumedAfterPanic(CoroutineKind),
    ResumedAfterDrop(CoroutineKind),
    MisalignedPointerDereference { required: Operand, found: Operand },
    NullPointerDereference,
    InvalidEnumConstruction(Operand),
}

impl AssertMessage {
    pub fn description(&self) -> Result<&'static str, Error> {
        match self {
            AssertMessage::Overflow(BinOp::Add, _, _) => Ok("attempt to add with overflow"),
            AssertMessage::Overflow(BinOp::Sub, _, _) => Ok("attempt to subtract with overflow"),
            AssertMessage::Overflow(BinOp::Mul, _, _) => Ok("attempt to multiply with overflow"),
            AssertMessage::Overflow(BinOp::Div, _, _) => Ok("attempt to divide with overflow"),
            AssertMessage::Overflow(BinOp::Rem, _, _) => {
                Ok("attempt to calculate the remainder with overflow")
            }
            AssertMessage::OverflowNeg(_) => Ok("attempt to negate with overflow"),
            AssertMessage::Overflow(BinOp::Shr, _, _) => Ok("attempt to shift right with overflow"),
            AssertMessage::Overflow(BinOp::Shl, _, _) => Ok("attempt to shift left with overflow"),
            AssertMessage::Overflow(op, _, _) => Err(error!("`{:?}` cannot overflow", op)),
            AssertMessage::DivisionByZero(_) => Ok("attempt to divide by zero"),
            AssertMessage::RemainderByZero(_) => {
                Ok("attempt to calculate the remainder with a divisor of zero")
            }
            AssertMessage::ResumedAfterReturn(CoroutineKind::Coroutine(_)) => {
                Ok("coroutine resumed after completion")
            }
            AssertMessage::ResumedAfterReturn(CoroutineKind::Desugared(
                CoroutineDesugaring::Async,
                _,
            )) => Ok("`async fn` resumed after completion"),
            AssertMessage::ResumedAfterReturn(CoroutineKind::Desugared(
                CoroutineDesugaring::Gen,
                _,
            )) => Ok("`async gen fn` resumed after completion"),
            AssertMessage::ResumedAfterReturn(CoroutineKind::Desugared(
                CoroutineDesugaring::AsyncGen,
                _,
            )) => Ok("`gen fn` should just keep returning `AssertMessage::None` after completion"),
            AssertMessage::ResumedAfterPanic(CoroutineKind::Coroutine(_)) => {
                Ok("coroutine resumed after panicking")
            }
            AssertMessage::ResumedAfterPanic(CoroutineKind::Desugared(
                CoroutineDesugaring::Async,
                _,
            )) => Ok("`async fn` resumed after panicking"),
            AssertMessage::ResumedAfterPanic(CoroutineKind::Desugared(
                CoroutineDesugaring::Gen,
                _,
            )) => Ok("`async gen fn` resumed after panicking"),
            AssertMessage::ResumedAfterPanic(CoroutineKind::Desugared(
                CoroutineDesugaring::AsyncGen,
                _,
            )) => Ok("`gen fn` should just keep returning `AssertMessage::None` after panicking"),

            AssertMessage::ResumedAfterDrop(CoroutineKind::Coroutine(_)) => {
                Ok("coroutine resumed after async drop")
            }
            AssertMessage::ResumedAfterDrop(CoroutineKind::Desugared(
                CoroutineDesugaring::Async,
                _,
            )) => Ok("`async fn` resumed after async drop"),
            AssertMessage::ResumedAfterDrop(CoroutineKind::Desugared(
                CoroutineDesugaring::Gen,
                _,
            )) => Ok("`async gen fn` resumed after async drop"),
            AssertMessage::ResumedAfterDrop(CoroutineKind::Desugared(
                CoroutineDesugaring::AsyncGen,
                _,
            )) => Ok("`gen fn` should just keep returning `AssertMessage::None` after async drop"),

            AssertMessage::BoundsCheck { .. } => Ok("index out of bounds"),
            AssertMessage::MisalignedPointerDereference { .. } => {
                Ok("misaligned pointer dereference")
            }
            AssertMessage::NullPointerDereference => Ok("null pointer dereference occurred"),
            AssertMessage::InvalidEnumConstruction(_) => {
                Ok("trying to construct an enum from an invalid value")
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum BinOp {
    Add,
    AddUnchecked,
    Sub,
    SubUnchecked,
    Mul,
    MulUnchecked,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    ShlUnchecked,
    Shr,
    ShrUnchecked,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Cmp,
    Offset,
}

impl BinOp {
    /// Return the type of this operation for the given input Ty.
    /// This function does not perform type checking, and it currently doesn't handle SIMD.
    pub fn ty(&self, lhs_ty: Ty, rhs_ty: Ty) -> Ty {
        with(|ctx| ctx.binop_ty(*self, lhs_ty, rhs_ty))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum UnOp {
    Not,
    Neg,
    PtrMetadata,
}

impl UnOp {
    /// Return the type of this operation for the given input Ty.
    /// This function does not perform type checking, and it currently doesn't handle SIMD.
    pub fn ty(&self, arg_ty: Ty) -> Ty {
        with(|ctx| ctx.unop_ty(*self, arg_ty))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum CoroutineKind {
    Desugared(CoroutineDesugaring, CoroutineSource),
    Coroutine(Movability),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum CoroutineSource {
    Block,
    Closure,
    Fn,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum CoroutineDesugaring {
    Async,

    Gen,

    AsyncGen,
}

pub(crate) type LocalDefId = Opaque;
/// The rustc coverage data structures are heavily tied to internal details of the
/// coverage implementation that are likely to change, and are unlikely to be
/// useful to third-party tools for the foreseeable future.
pub(crate) type Coverage = Opaque;

/// The FakeReadCause describes the type of pattern why a FakeRead statement exists.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum FakeReadCause {
    ForMatchGuard,
    ForMatchedPlace(LocalDefId),
    ForGuardBinding,
    ForLet(LocalDefId),
    ForIndex,
}

/// Describes what kind of retag is to be performed
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub enum RetagKind {
    FnEntry,
    TwoPhase,
    Raw,
    Default,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub enum Variance {
    Covariant,
    Invariant,
    Contravariant,
    Bivariant,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CopyNonOverlapping {
    pub src: Operand,
    pub dst: Operand,
    pub count: Operand,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum NonDivergingIntrinsic {
    Assume(Operand),
    CopyNonOverlapping(CopyNonOverlapping),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum StatementKind {
    Assign(Place, Rvalue),
    FakeRead(FakeReadCause, Place),
    SetDiscriminant { place: Place, variant_index: VariantIdx },
    Deinit(Place),
    StorageLive(Local),
    StorageDead(Local),
    Retag(RetagKind, Place),
    PlaceMention(Place),
    AscribeUserType { place: Place, projections: UserTypeProjection, variance: Variance },
    Coverage(Coverage),
    Intrinsic(NonDivergingIntrinsic),
    ConstEvalCounter,
    Nop,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum Rvalue {
    /// Creates a pointer with the indicated mutability to the place.
    ///
    /// This is generated by pointer casts like `&v as *const _` or raw address of expressions like
    /// `&raw v` or `addr_of!(v)`.
    AddressOf(RawPtrKind, Place),

    /// Creates an aggregate value, like a tuple or struct.
    ///
    /// This is needed because dataflow analysis needs to distinguish
    /// `dest = Foo { x: ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case that `Foo`
    /// has a destructor.
    ///
    /// Disallowed after deaggregation for all aggregate kinds except `Array` and `Coroutine`. After
    /// coroutine lowering, `Coroutine` aggregate kinds are disallowed too.
    Aggregate(AggregateKind, Vec<Operand>),

    /// * `Offset` has the same semantics as `<*const T>::offset`, except that the second
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
    BinaryOp(BinOp, Operand, Operand),

    /// Performs essentially all of the casts that can be performed via `as`.
    ///
    /// This allows for casts from/to a variety of types.
    Cast(CastKind, Operand, Ty),

    /// Same as `BinaryOp`, but yields `(T, bool)` with a `bool` indicating an error condition.
    ///
    /// For addition, subtraction, and multiplication on integers the error condition is set when
    /// the infinite precision result would not be equal to the actual result.
    CheckedBinaryOp(BinOp, Operand, Operand),

    /// A CopyForDeref is equivalent to a read from a place.
    /// When such a read happens, it is guaranteed that the only use of the returned value is a
    /// deref operation, immediately followed by one or more projections.
    CopyForDeref(Place),

    /// Computes the discriminant of the place, returning it as an integer.
    /// Returns zero for types without discriminant.
    ///
    /// The validity requirements for the underlying value are undecided for this rvalue, see
    /// [#91095]. Note too that the value of the discriminant is not the same thing as the
    /// variant index;
    ///
    /// [#91095]: https://github.com/rust-lang/rust/issues/91095
    Discriminant(Place),

    /// Yields the length of the place, as a `usize`.
    ///
    /// If the type of the place is an array, this is the array length. For slices (`[T]`, not
    /// `&[T]`) this accesses the place's metadata to determine the length. This rvalue is
    /// ill-formed for places of other types.
    Len(Place),

    /// Creates a reference to the place.
    Ref(Region, BorrowKind, Place),

    /// Creates an array where each element is the value of the operand.
    ///
    /// This is the cause of a bug in the case where the repetition count is zero because the value
    /// is not dropped, see [#74836].
    ///
    /// Corresponds to source code like `[x; 32]`.
    ///
    /// [#74836]: https://github.com/rust-lang/rust/issues/74836
    Repeat(Operand, TyConst),

    /// Transmutes a `*mut u8` into shallow-initialized `Box<T>`.
    ///
    /// This is different from a normal transmute because dataflow analysis will treat the box as
    /// initialized but its content as uninitialized. Like other pointer casts, this in general
    /// affects alias analysis.
    ShallowInitBox(Operand, Ty),

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
    ThreadLocalRef(stable_mir::CrateItem),

    /// Computes a value as described by the operation.
    NullaryOp(NullOp, Ty),

    /// Exactly like `BinaryOp`, but less operands.
    ///
    /// Also does two's-complement arithmetic. Negation requires a signed integer or a float;
    /// bitwise not requires a signed integer, unsigned integer, or bool. Both operation kinds
    /// return a value with the same type as their operand.
    UnaryOp(UnOp, Operand),

    /// Yields the operand unchanged
    Use(Operand),
}

impl Rvalue {
    pub fn ty(&self, locals: &[LocalDecl]) -> Result<Ty, Error> {
        match self {
            Rvalue::Use(operand) => operand.ty(locals),
            Rvalue::Repeat(operand, count) => {
                Ok(Ty::new_array_with_const_len(operand.ty(locals)?, count.clone()))
            }
            Rvalue::ThreadLocalRef(did) => Ok(did.ty()),
            Rvalue::Ref(reg, bk, place) => {
                let place_ty = place.ty(locals)?;
                Ok(Ty::new_ref(reg.clone(), place_ty, bk.to_mutable_lossy()))
            }
            Rvalue::AddressOf(mutability, place) => {
                let place_ty = place.ty(locals)?;
                Ok(Ty::new_ptr(place_ty, mutability.to_mutable_lossy()))
            }
            Rvalue::Len(..) => Ok(Ty::usize_ty()),
            Rvalue::Cast(.., ty) => Ok(*ty),
            Rvalue::BinaryOp(op, lhs, rhs) => {
                let lhs_ty = lhs.ty(locals)?;
                let rhs_ty = rhs.ty(locals)?;
                Ok(op.ty(lhs_ty, rhs_ty))
            }
            Rvalue::CheckedBinaryOp(op, lhs, rhs) => {
                let lhs_ty = lhs.ty(locals)?;
                let rhs_ty = rhs.ty(locals)?;
                let ty = op.ty(lhs_ty, rhs_ty);
                Ok(Ty::new_tuple(&[ty, Ty::bool_ty()]))
            }
            Rvalue::UnaryOp(op, operand) => {
                let arg_ty = operand.ty(locals)?;
                Ok(op.ty(arg_ty))
            }
            Rvalue::Discriminant(place) => {
                let place_ty = place.ty(locals)?;
                place_ty
                    .kind()
                    .discriminant_ty()
                    .ok_or_else(|| error!("Expected a `RigidTy` but found: {place_ty:?}"))
            }
            Rvalue::NullaryOp(NullOp::SizeOf | NullOp::AlignOf | NullOp::OffsetOf(..), _) => {
                Ok(Ty::usize_ty())
            }
            Rvalue::NullaryOp(NullOp::ContractChecks, _)
            | Rvalue::NullaryOp(NullOp::UbChecks, _) => Ok(Ty::bool_ty()),
            Rvalue::Aggregate(ak, ops) => match *ak {
                AggregateKind::Array(ty) => Ty::try_new_array(ty, ops.len() as u64),
                AggregateKind::Tuple => Ok(Ty::new_tuple(
                    &ops.iter().map(|op| op.ty(locals)).collect::<Result<Vec<_>, _>>()?,
                )),
                AggregateKind::Adt(def, _, ref args, _, _) => Ok(def.ty_with_args(args)),
                AggregateKind::Closure(def, ref args) => Ok(Ty::new_closure(def, args.clone())),
                AggregateKind::Coroutine(def, ref args, mov) => {
                    Ok(Ty::new_coroutine(def, args.clone(), mov))
                }
                AggregateKind::CoroutineClosure(def, ref args) => {
                    Ok(Ty::new_coroutine_closure(def, args.clone()))
                }
                AggregateKind::RawPtr(ty, mutability) => Ok(Ty::new_ptr(ty, mutability)),
            },
            Rvalue::ShallowInitBox(_, ty) => Ok(Ty::new_box(*ty)),
            Rvalue::CopyForDeref(place) => place.ty(locals),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum AggregateKind {
    Array(Ty),
    Tuple,
    Adt(AdtDef, VariantIdx, GenericArgs, Option<UserTypeAnnotationIndex>, Option<FieldIdx>),
    Closure(ClosureDef, GenericArgs),
    // FIXME(stable_mir): Movability here is redundant
    Coroutine(CoroutineDef, GenericArgs, Movability),
    CoroutineClosure(CoroutineClosureDef, GenericArgs),
    RawPtr(Ty, Mutability),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(ConstOperand),
}

#[derive(Clone, Eq, PartialEq, Serialize)]
pub struct Place {
    pub local: Local,
    /// projection out of a place (access a field, deref a pointer, etc)
    pub projection: Vec<ProjectionElem>,
}

impl From<Local> for Place {
    fn from(local: Local) -> Self {
        Place { local, projection: vec![] }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConstOperand {
    pub span: Span,
    pub user_ty: Option<UserTypeAnnotationIndex>,
    pub const_: MirConst,
}

/// Debug information pertaining to a user variable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VarDebugInfo {
    /// The variable name.
    pub name: Symbol,

    /// Source info of the user variable, including the scope
    /// within which the variable is visible (to debuginfo).
    pub source_info: SourceInfo,

    /// The user variable's data is split across several fragments,
    /// each described by a `VarDebugInfoFragment`.
    pub composite: Option<VarDebugInfoFragment>,

    /// Where the data for this user variable is to be found.
    pub value: VarDebugInfoContents,

    /// When present, indicates what argument number this variable is in the function that it
    /// originated from (starting from 1). Note, if MIR inlining is enabled, then this is the
    /// argument number in the original function before it was inlined.
    pub argument_index: Option<u16>,
}

impl VarDebugInfo {
    /// Return a local variable if this info is related to one.
    pub fn local(&self) -> Option<Local> {
        match &self.value {
            VarDebugInfoContents::Place(place) if place.projection.is_empty() => Some(place.local),
            VarDebugInfoContents::Place(_) | VarDebugInfoContents::Const(_) => None,
        }
    }

    /// Return a constant if this info is related to one.
    pub fn constant(&self) -> Option<&ConstOperand> {
        match &self.value {
            VarDebugInfoContents::Place(_) => None,
            VarDebugInfoContents::Const(const_op) => Some(const_op),
        }
    }
}

pub type SourceScope = u32;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceInfo {
    pub span: Span,
    pub scope: SourceScope,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VarDebugInfoFragment {
    pub ty: Ty,
    pub projection: Vec<ProjectionElem>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum VarDebugInfoContents {
    Place(Place),
    Const(ConstOperand),
}

// In MIR ProjectionElem is parameterized on the second Field argument and the Index argument. This
// is so it can be used for both Places (for which the projection elements are of type
// ProjectionElem<Local, Ty>) and user-provided type annotations (for which the projection elements
// are of type ProjectionElem<(), ()>). In SMIR we don't need this generality, so we just use
// ProjectionElem for Places.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ProjectionElem {
    /// Dereference projections (e.g. `*_1`) project to the address referenced by the base place.
    Deref,

    /// A field projection (e.g., `f` in `_1.f`) project to a field in the base place. The field is
    /// referenced by source-order index rather than the name of the field. The fields type is also
    /// given.
    Field(FieldIdx, Ty),

    /// Index into a slice/array. The value of the index is computed at runtime using the `V`
    /// argument.
    ///
    /// Note that this does not also dereference, and so it does not exactly correspond to slice
    /// indexing in Rust. In other words, in the below Rust code:
    ///
    /// ```rust
    /// let x = &[1, 2, 3, 4];
    /// let i = 2;
    /// x[i];
    /// ```
    ///
    /// The `x[i]` is turned into a `Deref` followed by an `Index`, not just an `Index`. The same
    /// thing is true of the `ConstantIndex` and `Subslice` projections below.
    Index(Local),

    /// Index into a slice/array given by offsets.
    ///
    /// These indices are generated by slice patterns. Easiest to explain by example:
    ///
    /// ```ignore (illustrative)
    /// [X, _, .._, _, _] => { offset: 0, min_length: 4, from_end: false },
    /// [_, X, .._, _, _] => { offset: 1, min_length: 4, from_end: false },
    /// [_, _, .._, X, _] => { offset: 2, min_length: 4, from_end: true },
    /// [_, _, .._, _, X] => { offset: 1, min_length: 4, from_end: true },
    /// ```
    ConstantIndex {
        /// index or -index (in Python terms), depending on from_end
        offset: u64,
        /// The thing being indexed must be at least this long -- otherwise, the
        /// projection is UB.
        ///
        /// For arrays this is always the exact length.
        min_length: u64,
        /// Counting backwards from end? This is always false when indexing an
        /// array.
        from_end: bool,
    },

    /// Projects a slice from the base place.
    ///
    /// These indices are generated by slice patterns. If `from_end` is true, this represents
    /// `slice[from..slice.len() - to]`. Otherwise it represents `array[from..to]`.
    Subslice {
        from: u64,
        to: u64,
        /// Whether `to` counts from the start or end of the array/slice.
        from_end: bool,
    },

    /// "Downcast" to a variant of an enum or a coroutine.
    Downcast(VariantIdx),

    /// Like an explicit cast from an opaque type to a concrete type, but without
    /// requiring an intermediate variable.
    OpaqueCast(Ty),

    /// A `Subtype(T)` projection is applied to any `StatementKind::Assign` where
    /// type of lvalue doesn't match the type of rvalue, the primary goal is making subtyping
    /// explicit during optimizations and codegen.
    ///
    /// This projection doesn't impact the runtime behavior of the program except for potentially changing
    /// some type metadata of the interpreter or codegen backend.
    Subtype(Ty),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UserTypeProjection {
    pub base: UserTypeAnnotationIndex,

    pub projection: Opaque,
}

pub type Local = usize;

pub const RETURN_LOCAL: Local = 0;

/// The source-order index of a field in a variant.
///
/// For example, in the following types,
/// ```ignore(illustrative)
/// enum Demo1 {
///    Variant0 { a: bool, b: i32 },
///    Variant1 { c: u8, d: u64 },
/// }
/// struct Demo2 { e: u8, f: u16, g: u8 }
/// ```
/// `a`'s `FieldIdx` is `0`,
/// `b`'s `FieldIdx` is `1`,
/// `c`'s `FieldIdx` is `0`, and
/// `g`'s `FieldIdx` is `2`.
pub type FieldIdx = usize;

type UserTypeAnnotationIndex = usize;

/// The possible branch sites of a [TerminatorKind::SwitchInt].
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SwitchTargets {
    /// The conditional branches where the first element represents the value that guards this
    /// branch, and the second element is the branch target.
    branches: Vec<(u128, BasicBlockIdx)>,
    /// The `otherwise` branch which will be taken in case none of the conditional branches are
    /// satisfied.
    otherwise: BasicBlockIdx,
}

impl SwitchTargets {
    /// All possible targets including the `otherwise` target.
    pub fn all_targets(&self) -> Successors {
        self.branches.iter().map(|(_, target)| *target).chain(Some(self.otherwise)).collect()
    }

    /// The `otherwise` branch target.
    pub fn otherwise(&self) -> BasicBlockIdx {
        self.otherwise
    }

    /// The conditional targets which are only taken if the pattern matches the given value.
    pub fn branches(&self) -> impl Iterator<Item = (u128, BasicBlockIdx)> {
        self.branches.iter().copied()
    }

    /// The number of targets including `otherwise`.
    pub fn len(&self) -> usize {
        self.branches.len() + 1
    }

    /// Create a new SwitchTargets from the given branches and `otherwise` target.
    pub fn new(branches: Vec<(u128, BasicBlockIdx)>, otherwise: BasicBlockIdx) -> SwitchTargets {
        SwitchTargets { branches, otherwise }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Shared,

    /// An immutable, aliasable borrow that is discarded after borrow-checking. Can behave either
    /// like a normal shared borrow or like a special shallow borrow (see [`FakeBorrowKind`]).
    Fake(FakeBorrowKind),

    /// Data is mutable and not aliasable.
    Mut {
        /// `true` if this borrow arose from method-call auto-ref
        kind: MutBorrowKind,
    },
}

impl BorrowKind {
    pub fn to_mutable_lossy(self) -> Mutability {
        match self {
            BorrowKind::Mut { .. } => Mutability::Mut,
            BorrowKind::Shared => Mutability::Not,
            // FIXME: There's no type corresponding to a shallow borrow, so use `&` as an approximation.
            BorrowKind::Fake(_) => Mutability::Not,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum RawPtrKind {
    Mut,
    Const,
    FakeForPtrMetadata,
}

impl RawPtrKind {
    pub fn to_mutable_lossy(self) -> Mutability {
        match self {
            RawPtrKind::Mut { .. } => Mutability::Mut,
            RawPtrKind::Const => Mutability::Not,
            // FIXME: There's no type corresponding to a shallow borrow, so use `&` as an approximation.
            RawPtrKind::FakeForPtrMetadata => Mutability::Not,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum MutBorrowKind {
    Default,
    TwoPhaseBorrow,
    ClosureCapture,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum FakeBorrowKind {
    /// A shared (deep) borrow. Data must be immutable and is aliasable.
    Deep,
    /// The immediately borrowed place must be immutable, but projections from
    /// it don't need to be. This is used to prevent match guards from replacing
    /// the scrutinee. For example, a fake borrow of `a.b` doesn't
    /// conflict with a mutable borrow of `a.b.c`.
    Shallow,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Mutability {
    Not,
    Mut,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum Safety {
    Safe,
    Unsafe,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum PointerCoercion {
    /// Go from a fn-item type to a fn-pointer type.
    ReifyFnPointer,

    /// Go from a safe fn pointer to an unsafe fn pointer.
    UnsafeFnPointer,

    /// Go from a non-capturing closure to a fn pointer or an unsafe fn pointer.
    /// It cannot convert a closure that requires unsafe.
    ClosureFnPointer(Safety),

    /// Go from a mut raw pointer to a const raw pointer.
    MutToConstPointer,

    /// Go from `*const [T; N]` to `*const T`
    ArrayToPointer,

    /// Unsize a pointer/reference value, e.g., `&[T; n]` to
    /// `&[T]`. Note that the source could be a thin or wide pointer.
    /// This will do things like convert thin pointers to wide
    /// pointers, or convert structs containing thin pointers to
    /// structs containing wide pointers, or convert between wide
    /// pointers.
    Unsize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum CastKind {
    // FIXME(smir-rename): rename this to PointerExposeProvenance
    PointerExposeAddress,
    PointerWithExposedProvenance,
    PointerCoercion(PointerCoercion),
    IntToInt,
    FloatToInt,
    FloatToFloat,
    IntToFloat,
    PtrToPtr,
    FnPtrToPtr,
    Transmute,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum NullOp {
    /// Returns the size of a value of that type.
    SizeOf,
    /// Returns the minimum alignment of a type.
    AlignOf,
    /// Returns the offset of a field.
    OffsetOf(Vec<(VariantIdx, FieldIdx)>),
    /// cfg!(ub_checks), but at codegen time
    UbChecks,
    /// cfg!(contract_checks), but at codegen time
    ContractChecks,
}

impl Operand {
    /// Get the type of an operand relative to the local declaration.
    ///
    /// In order to retrieve the correct type, the `locals` argument must match the list of all
    /// locals from the function body where this operand originates from.
    ///
    /// Errors indicate a malformed operand or incompatible locals list.
    pub fn ty(&self, locals: &[LocalDecl]) -> Result<Ty, Error> {
        match self {
            Operand::Copy(place) | Operand::Move(place) => place.ty(locals),
            Operand::Constant(c) => Ok(c.ty()),
        }
    }
}

impl ConstOperand {
    pub fn ty(&self) -> Ty {
        self.const_.ty()
    }
}

impl Place {
    /// Resolve down the chain of projections to get the type referenced at the end of it.
    /// E.g.:
    /// Calling `ty()` on `var.field` should return the type of `field`.
    ///
    /// In order to retrieve the correct type, the `locals` argument must match the list of all
    /// locals from the function body where this place originates from.
    pub fn ty(&self, locals: &[LocalDecl]) -> Result<Ty, Error> {
        self.projection.iter().try_fold(locals[self.local].ty, |place_ty, elem| elem.ty(place_ty))
    }
}

impl ProjectionElem {
    /// Get the expected type after applying this projection to a given place type.
    pub fn ty(&self, place_ty: Ty) -> Result<Ty, Error> {
        let ty = place_ty;
        match &self {
            ProjectionElem::Deref => Self::deref_ty(ty),
            ProjectionElem::Field(_idx, fty) => Ok(*fty),
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => Self::index_ty(ty),
            ProjectionElem::Subslice { from, to, from_end } => {
                Self::subslice_ty(ty, *from, *to, *from_end)
            }
            ProjectionElem::Downcast(_) => Ok(ty),
            ProjectionElem::OpaqueCast(ty) | ProjectionElem::Subtype(ty) => Ok(*ty),
        }
    }

    fn index_ty(ty: Ty) -> Result<Ty, Error> {
        ty.kind().builtin_index().ok_or_else(|| error!("Cannot index non-array type: {ty:?}"))
    }

    fn subslice_ty(ty: Ty, from: u64, to: u64, from_end: bool) -> Result<Ty, Error> {
        let ty_kind = ty.kind();
        match ty_kind {
            TyKind::RigidTy(RigidTy::Slice(..)) => Ok(ty),
            TyKind::RigidTy(RigidTy::Array(inner, _)) if !from_end => Ty::try_new_array(
                inner,
                to.checked_sub(from).ok_or_else(|| error!("Subslice overflow: {from}..{to}"))?,
            ),
            TyKind::RigidTy(RigidTy::Array(inner, size)) => {
                let size = size.eval_target_usize()?;
                let len = size - from - to;
                Ty::try_new_array(inner, len)
            }
            _ => Err(Error(format!("Cannot subslice non-array type: `{ty_kind:?}`"))),
        }
    }

    fn deref_ty(ty: Ty) -> Result<Ty, Error> {
        let deref_ty = ty
            .kind()
            .builtin_deref(true)
            .ok_or_else(|| error!("Cannot dereference type: {ty:?}"))?;
        Ok(deref_ty.ty)
    }
}
