use crate::mir::pretty::{function_body, pretty_statement};
use crate::ty::{AdtDef, ClosureDef, Const, CoroutineDef, GenericArgs, Movability, Region, Ty};
use crate::Opaque;
use crate::Span;
use std::io;
/// The SMIR representation of a single function.
#[derive(Clone, Debug)]
pub struct Body {
    pub blocks: Vec<BasicBlock>,

    // Declarations of locals within the function.
    //
    // The first local is the return value pointer, followed by `arg_count`
    // locals for the function arguments, followed by any user-declared
    // variables and temporaries.
    pub(super) locals: LocalDecls,

    // The number of arguments this function takes.
    pub(super) arg_count: usize,
}

impl Body {
    /// Constructs a `Body`.
    ///
    /// A constructor is required to build a `Body` from outside the crate
    /// because the `arg_count` and `locals` fields are private.
    pub fn new(blocks: Vec<BasicBlock>, locals: LocalDecls, arg_count: usize) -> Self {
        // If locals doesn't contain enough entries, it can lead to panics in
        // `ret_local`, `arg_locals`, and `inner_locals`.
        assert!(
            locals.len() > arg_count,
            "A Body must contain at least a local for the return value and each of the function's arguments"
        );
        Self { blocks, locals, arg_count }
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

    /// Convenience function to get all the locals in this function.
    ///
    /// Locals are typically accessed via the more specific methods `ret_local`,
    /// `arg_locals`, and `inner_locals`.
    pub fn locals(&self) -> &[LocalDecl] {
        &self.locals
    }

    pub fn dump<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "{}", function_body(self))?;
        self.blocks
            .iter()
            .enumerate()
            .map(|(index, block)| -> io::Result<()> {
                writeln!(w, "    bb{}: {{", index)?;
                let _ = block
                    .statements
                    .iter()
                    .map(|statement| -> io::Result<()> {
                        writeln!(w, "{}", pretty_statement(&statement.kind))?;
                        Ok(())
                    })
                    .collect::<Vec<_>>();
                writeln!(w, "    }}").unwrap();
                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}

type LocalDecls = Vec<LocalDecl>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalDecl {
    pub ty: Ty,
    pub span: Span,
    pub mutability: Mutability,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Terminator {
    pub kind: TerminatorKind,
    pub span: Span,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminatorKind {
    Goto {
        target: usize,
    },
    SwitchInt {
        discr: Operand,
        targets: Vec<SwitchTarget>,
        otherwise: usize,
    },
    Resume,
    Abort,
    Return,
    Unreachable,
    Drop {
        place: Place,
        target: usize,
        unwind: UnwindAction,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<usize>,
        unwind: UnwindAction,
    },
    Assert {
        cond: Operand,
        expected: bool,
        msg: AssertMessage,
        target: usize,
        unwind: UnwindAction,
    },
    CoroutineDrop,
    InlineAsm {
        template: String,
        operands: Vec<InlineAsmOperand>,
        options: String,
        line_spans: String,
        destination: Option<usize>,
        unwind: UnwindAction,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InlineAsmOperand {
    pub in_value: Option<Operand>,
    pub out_place: Option<Place>,
    // This field has a raw debug representation of MIR's InlineAsmOperand.
    // For now we care about place/operand + the rest in a debug format.
    pub raw_rpr: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnwindAction {
    Continue,
    Unreachable,
    Terminate,
    Cleanup(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssertMessage {
    BoundsCheck { len: Operand, index: Operand },
    Overflow(BinOp, Operand, Operand),
    OverflowNeg(Operand),
    DivisionByZero(Operand),
    RemainderByZero(Operand),
    ResumedAfterReturn(CoroutineKind),
    ResumedAfterPanic(CoroutineKind),
    MisalignedPointerDereference { required: Operand, found: Operand },
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
    Offset,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnOp {
    Not,
    Neg,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoroutineKind {
    Async(CoroutineSource),
    Coroutine,
    Gen(CoroutineSource),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoroutineSource {
    Block,
    Closure,
    Fn,
}

pub(crate) type LocalDefId = Opaque;
/// The rustc coverage data structures are heavily tied to internal details of the
/// coverage implementation that are likely to change, and are unlikely to be
/// useful to third-party tools for the foreseeable future.
pub(crate) type Coverage = Opaque;

/// The FakeReadCause describes the type of pattern why a FakeRead statement exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FakeReadCause {
    ForMatchGuard,
    ForMatchedPlace(LocalDefId),
    ForGuardBinding,
    ForLet(LocalDefId),
    ForIndex,
}

/// Describes what kind of retag is to be performed
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum RetagKind {
    FnEntry,
    TwoPhase,
    Raw,
    Default,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Variance {
    Covariant,
    Invariant,
    Contravariant,
    Bivariant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CopyNonOverlapping {
    pub src: Operand,
    pub dst: Operand,
    pub count: Operand,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NonDivergingIntrinsic {
    Assume(Operand),
    CopyNonOverlapping(CopyNonOverlapping),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rvalue {
    /// Creates a pointer with the indicated mutability to the place.
    ///
    /// This is generated by pointer casts like `&v as *const _` or raw address of expressions like
    /// `&raw v` or `addr_of!(v)`.
    AddressOf(Mutability, Place),

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
    Repeat(Operand, Const),

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
    ThreadLocalRef(crate::CrateItem),

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AggregateKind {
    Array(Ty),
    Tuple,
    Adt(AdtDef, VariantIdx, GenericArgs, Option<UserTypeAnnotationIndex>, Option<FieldIdx>),
    Closure(ClosureDef, GenericArgs),
    Coroutine(CoroutineDef, GenericArgs, Movability),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(Constant),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Place {
    pub local: Local,
    /// projection out of a place (access a field, deref a pointer, etc)
    pub projection: Vec<ProjectionElem>,
}

// In MIR ProjectionElem is parameterized on the second Field argument and the Index argument. This
// is so it can be used for both Places (for which the projection elements are of type
// ProjectionElem<Local, Ty>) and user-provided type annotations (for which the projection elements
// are of type ProjectionElem<(), ()>). In SMIR we don't need this generality, so we just use
// ProjectionElem for Places.
#[derive(Clone, Debug, Eq, PartialEq)]
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
        /// The thing being indexed must be at least this long. For arrays this
        /// is always the exact length.
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
type FieldIdx = usize;

/// The source-order index of a variant in a type.
///
/// For example, in the following types,
/// ```ignore(illustrative)
/// enum Demo1 {
///    Variant0 { a: bool, b: i32 },
///    Variant1 { c: u8, d: u64 },
/// }
/// struct Demo2 { e: u8, f: u16, g: u8 }
/// ```
/// `a` is in the variant with the `VariantIdx` of `0`,
/// `c` is in the variant with the `VariantIdx` of `1`, and
/// `g` is in the variant with the `VariantIdx` of `0`.
pub type VariantIdx = usize;

type UserTypeAnnotationIndex = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Constant {
    pub span: Span,
    pub user_ty: Option<UserTypeAnnotationIndex>,
    pub literal: Const,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SwitchTarget {
    pub value: u128,
    pub target: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Shared,

    /// The immediately borrowed place must be immutable, but projections from
    /// it don't need to be. This is used to prevent match guards from replacing
    /// the scrutinee. For example, a fake borrow of `a.b` doesn't
    /// conflict with a mutable borrow of `a.b.c`.
    Fake,

    /// Data is mutable and not aliasable.
    Mut {
        /// `true` if this borrow arose from method-call auto-ref
        kind: MutBorrowKind,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MutBorrowKind {
    Default,
    TwoPhaseBorrow,
    ClosureCapture,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mutability {
    Not,
    Mut,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Safety {
    Unsafe,
    Normal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PointerCoercion {
    /// Go from a fn-item type to a fn-pointer type.
    ReifyFnPointer,

    /// Go from a safe fn pointer to an unsafe fn pointer.
    UnsafeFnPointer,

    /// Go from a non-capturing closure to an fn pointer or an unsafe fn pointer.
    /// It cannot convert a closure that requires unsafe.
    ClosureFnPointer(Safety),

    /// Go from a mut raw pointer to a const raw pointer.
    MutToConstPointer,

    /// Go from `*const [T; N]` to `*const T`
    ArrayToPointer,

    /// Unsize a pointer/reference value, e.g., `&[T; n]` to
    /// `&[T]`. Note that the source could be a thin or fat pointer.
    /// This will do things like convert thin pointers to fat
    /// pointers, or convert structs containing thin pointers to
    /// structs containing fat pointers, or convert between fat
    /// pointers.
    Unsize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CastKind {
    PointerExposeAddress,
    PointerFromExposedAddress,
    PointerCoercion(PointerCoercion),
    DynStar,
    IntToInt,
    FloatToInt,
    FloatToFloat,
    IntToFloat,
    PtrToPtr,
    FnPtrToPtr,
    Transmute,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NullOp {
    /// Returns the size of a value of that type.
    SizeOf,
    /// Returns the minimum alignment of a type.
    AlignOf,
    /// Returns the offset of a field.
    OffsetOf(Vec<(VariantIdx, FieldIdx)>),
}

impl Operand {
    pub fn ty(&self, locals: &[LocalDecl]) -> Ty {
        match self {
            Operand::Copy(place) | Operand::Move(place) => place.ty(locals),
            Operand::Constant(c) => c.ty(),
        }
    }
}

impl Constant {
    pub fn ty(&self) -> Ty {
        self.literal.ty()
    }
}

impl Place {
    // FIXME(klinvill): This function is expected to resolve down the chain of projections to get
    // the type referenced at the end of it. E.g. calling `ty()` on `*(_1.f)` should end up
    // returning the type referenced by `f`. The information needed to do this may not currently be
    // present in Stable MIR since at least an implementation for AdtDef is probably needed.
    pub fn ty(&self, locals: &[LocalDecl]) -> Ty {
        let _start_ty = locals[self.local].ty;
        todo!("Implement projection")
    }
}
