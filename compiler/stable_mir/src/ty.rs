use super::{
    mir::Safety,
    mir::{Body, Mutability},
    with, AllocId, DefId, Symbol,
};
use crate::{Filename, Opaque};
use std::fmt::{self, Debug, Formatter};

#[derive(Copy, Clone)]
pub struct Ty(pub usize);

impl Debug for Ty {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ty").field("id", &self.0).field("kind", &self.kind()).finish()
    }
}

impl Ty {
    pub fn kind(&self) -> TyKind {
        with(|context| context.ty_kind(*self))
    }
}

impl From<TyKind> for Ty {
    fn from(value: TyKind) -> Self {
        with(|context| context.mk_ty(value))
    }
}

#[derive(Debug, Clone)]
pub struct Const {
    pub literal: ConstantKind,
    pub ty: Ty,
}

type Ident = Opaque;

#[derive(Debug, Clone)]
pub struct Region {
    pub kind: RegionKind,
}

#[derive(Debug, Clone)]
pub enum RegionKind {
    ReEarlyBound(EarlyBoundRegion),
    ReLateBound(DebruijnIndex, BoundRegion),
    ReStatic,
    RePlaceholder(Placeholder<BoundRegion>),
    ReErased,
}

pub(crate) type DebruijnIndex = u32;

#[derive(Debug, Clone)]
pub struct EarlyBoundRegion {
    pub def_id: RegionDef,
    pub index: u32,
    pub name: Symbol,
}

pub(crate) type BoundVar = u32;

#[derive(Debug, Clone)]
pub struct BoundRegion {
    pub var: BoundVar,
    pub kind: BoundRegionKind,
}

pub(crate) type UniverseIndex = u32;

#[derive(Debug, Clone)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub bound: T,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Span(usize);

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Span")
            .field("id", &self.0)
            .field("repr", &with(|cx| cx.span_to_string(*self)))
            .finish()
    }
}

impl Span {
    /// Return filename for diagnostic purposes
    pub fn get_filename(&self) -> Filename {
        with(|c| c.get_filename(self))
    }

    /// Return lines that corespond to this `Span`
    pub fn get_lines(&self) -> LineInfo {
        with(|c| c.get_lines(&self))
    }
}

#[derive(Clone, Copy, Debug)]
/// Information you get from `Span` in a struct form.
/// Line and col start from 1.
pub struct LineInfo {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl IndexedVal for Span {
    fn to_val(index: usize) -> Self {
        Span(index)
    }
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub enum TyKind {
    RigidTy(RigidTy),
    Alias(AliasKind, AliasTy),
    Param(ParamTy),
    Bound(usize, BoundTy),
}

#[derive(Clone, Debug)]
pub enum RigidTy {
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Adt(AdtDef, GenericArgs),
    Foreign(ForeignDef),
    Str,
    Array(Ty, Const),
    Slice(Ty),
    RawPtr(Ty, Mutability),
    Ref(Region, Ty, Mutability),
    FnDef(FnDef, GenericArgs),
    FnPtr(PolyFnSig),
    Closure(ClosureDef, GenericArgs),
    Coroutine(CoroutineDef, GenericArgs, Movability),
    Dynamic(Vec<Binder<ExistentialPredicate>>, Region, DynKind),
    Never,
    Tuple(Vec<Ty>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTy {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ForeignDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FnDef(pub DefId);

impl FnDef {
    pub fn body(&self) -> Body {
        with(|ctx| ctx.mir_body(self.0))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClosureDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CoroutineDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BrNamedDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AdtDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AliasDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TraitDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GenericDef(pub DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConstDef(pub DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ImplDef(pub DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RegionDef(pub DefId);

/// A list of generic arguments.
#[derive(Clone, Debug)]
pub struct GenericArgs(pub Vec<GenericArgKind>);

impl std::ops::Index<ParamTy> for GenericArgs {
    type Output = Ty;

    fn index(&self, index: ParamTy) -> &Self::Output {
        self.0[index.index as usize].expect_ty()
    }
}

impl std::ops::Index<ParamConst> for GenericArgs {
    type Output = Const;

    fn index(&self, index: ParamConst) -> &Self::Output {
        self.0[index.index as usize].expect_const()
    }
}

#[derive(Clone, Debug)]
pub enum GenericArgKind {
    Lifetime(Region),
    Type(Ty),
    Const(Const),
}

impl GenericArgKind {
    /// Panic if this generic argument is not a type, otherwise
    /// return the type.
    #[track_caller]
    pub fn expect_ty(&self) -> &Ty {
        match self {
            GenericArgKind::Type(ty) => ty,
            _ => panic!("{self:?}"),
        }
    }

    /// Panic if this generic argument is not a const, otherwise
    /// return the const.
    #[track_caller]
    pub fn expect_const(&self) -> &Const {
        match self {
            GenericArgKind::Const(c) => c,
            _ => panic!("{self:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum TermKind {
    Type(Ty),
    Const(Const),
}

#[derive(Clone, Debug)]
pub enum AliasKind {
    Projection,
    Inherent,
    Opaque,
    Weak,
}

#[derive(Clone, Debug)]
pub struct AliasTy {
    pub def_id: AliasDef,
    pub args: GenericArgs,
}

pub type PolyFnSig = Binder<FnSig>;

#[derive(Clone, Debug)]
pub struct FnSig {
    pub inputs_and_output: Vec<Ty>,
    pub c_variadic: bool,
    pub unsafety: Safety,
    pub abi: Abi,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Abi {
    Rust,
    C { unwind: bool },
    Cdecl { unwind: bool },
    Stdcall { unwind: bool },
    Fastcall { unwind: bool },
    Vectorcall { unwind: bool },
    Thiscall { unwind: bool },
    Aapcs { unwind: bool },
    Win64 { unwind: bool },
    SysV64 { unwind: bool },
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,
    EfiApi,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    CCmseNonSecureCall,
    Wasm,
    System { unwind: bool },
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
    Unadjusted,
    RustCold,
    RiscvInterruptM,
    RiscvInterruptS,
}

#[derive(Clone, Debug)]
pub struct Binder<T> {
    pub value: T,
    pub bound_vars: Vec<BoundVariableKind>,
}

#[derive(Clone, Debug)]
pub struct EarlyBinder<T> {
    pub value: T,
}

#[derive(Clone, Debug)]
pub enum BoundVariableKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BoundTyKind {
    Anon,
    Param(ParamDef, String),
}

#[derive(Clone, Debug)]
pub enum BoundRegionKind {
    BrAnon,
    BrNamed(BrNamedDef, String),
    BrEnv,
}

#[derive(Clone, Debug)]
pub enum DynKind {
    Dyn,
    DynStar,
}

#[derive(Clone, Debug)]
pub enum ExistentialPredicate {
    Trait(ExistentialTraitRef),
    Projection(ExistentialProjection),
    AutoTrait(TraitDef),
}

#[derive(Clone, Debug)]
pub struct ExistentialTraitRef {
    pub def_id: TraitDef,
    pub generic_args: GenericArgs,
}

#[derive(Clone, Debug)]
pub struct ExistentialProjection {
    pub def_id: TraitDef,
    pub generic_args: GenericArgs,
    pub term: TermKind,
}

#[derive(Clone, Debug)]
pub struct ParamTy {
    pub index: u32,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct BoundTy {
    pub var: usize,
    pub kind: BoundTyKind,
}

pub type Bytes = Vec<Option<u8>>;
pub type Size = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prov(pub AllocId);
pub type Align = u64;
pub type Promoted = u32;
pub type InitMaskMaterialized = Vec<u64>;

/// Stores the provenance information of pointers stored in memory.
#[derive(Clone, Debug)]
pub struct ProvenanceMap {
    /// Provenance in this map applies from the given offset for an entire pointer-size worth of
    /// bytes. Two entries in this map are always at least a pointer size apart.
    pub ptrs: Vec<(Size, Prov)>,
}

#[derive(Clone, Debug)]
pub struct Allocation {
    pub bytes: Bytes,
    pub provenance: ProvenanceMap,
    pub align: Align,
    pub mutability: Mutability,
}

#[derive(Clone, Debug)]
pub enum ConstantKind {
    Allocated(Allocation),
    Unevaluated(UnevaluatedConst),
    Param(ParamConst),
}

#[derive(Clone, Debug)]
pub struct ParamConst {
    pub index: u32,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct UnevaluatedConst {
    pub def: ConstDef,
    pub args: GenericArgs,
    pub promoted: Option<Promoted>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraitSpecializationKind {
    None,
    Marker,
    AlwaysApplicable,
}

#[derive(Clone, Debug)]
pub struct TraitDecl {
    pub def_id: TraitDef,
    pub unsafety: Safety,
    pub paren_sugar: bool,
    pub has_auto_impl: bool,
    pub is_marker: bool,
    pub is_coinductive: bool,
    pub skip_array_during_method_dispatch: bool,
    pub specialization_kind: TraitSpecializationKind,
    pub must_implement_one_of: Option<Vec<Ident>>,
    pub implement_via_object: bool,
    pub deny_explicit_impl: bool,
}

impl TraitDecl {
    pub fn generics_of(&self) -> Generics {
        with(|cx| cx.generics_of(self.def_id.0))
    }

    pub fn predicates_of(&self) -> GenericPredicates {
        with(|cx| cx.predicates_of(self.def_id.0))
    }

    pub fn explicit_predicates_of(&self) -> GenericPredicates {
        with(|cx| cx.explicit_predicates_of(self.def_id.0))
    }
}

pub type ImplTrait = EarlyBinder<TraitRef>;

#[derive(Clone, Debug)]
pub struct TraitRef {
    pub def_id: TraitDef,
    pub args: GenericArgs,
}

#[derive(Clone, Debug)]
pub struct Generics {
    pub parent: Option<GenericDef>,
    pub parent_count: usize,
    pub params: Vec<GenericParamDef>,
    pub param_def_id_to_index: Vec<(GenericDef, u32)>,
    pub has_self: bool,
    pub has_late_bound_regions: Option<Span>,
    pub host_effect_index: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum GenericParamDefKind {
    Lifetime,
    Type { has_default: bool, synthetic: bool },
    Const { has_default: bool },
}

#[derive(Clone, Debug)]
pub struct GenericParamDef {
    pub name: super::Symbol,
    pub def_id: GenericDef,
    pub index: u32,
    pub pure_wrt_drop: bool,
    pub kind: GenericParamDefKind,
}

pub struct GenericPredicates {
    pub parent: Option<TraitDef>,
    pub predicates: Vec<(PredicateKind, Span)>,
}

#[derive(Clone, Debug)]
pub enum PredicateKind {
    Clause(ClauseKind),
    ObjectSafe(TraitDef),
    ClosureKind(ClosureDef, GenericArgs, ClosureKind),
    SubType(SubtypePredicate),
    Coerce(CoercePredicate),
    ConstEquate(Const, Const),
    Ambiguous,
    AliasRelate(TermKind, TermKind, AliasRelationDirection),
}

#[derive(Clone, Debug)]
pub enum ClauseKind {
    Trait(TraitPredicate),
    RegionOutlives(RegionOutlivesPredicate),
    TypeOutlives(TypeOutlivesPredicate),
    Projection(ProjectionPredicate),
    ConstArgHasType(Const, Ty),
    WellFormed(GenericArgKind),
    ConstEvaluatable(Const),
}

#[derive(Clone, Debug)]
pub enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

#[derive(Clone, Debug)]
pub struct SubtypePredicate {
    pub a: Ty,
    pub b: Ty,
}

#[derive(Clone, Debug)]
pub struct CoercePredicate {
    pub a: Ty,
    pub b: Ty,
}

#[derive(Clone, Debug)]
pub enum AliasRelationDirection {
    Equate,
    Subtype,
}

#[derive(Clone, Debug)]
pub struct TraitPredicate {
    pub trait_ref: TraitRef,
    pub polarity: ImplPolarity,
}

#[derive(Clone, Debug)]
pub struct OutlivesPredicate<A, B>(pub A, pub B);

pub type RegionOutlivesPredicate = OutlivesPredicate<Region, Region>;
pub type TypeOutlivesPredicate = OutlivesPredicate<Ty, Region>;

#[derive(Clone, Debug)]
pub struct ProjectionPredicate {
    pub projection_ty: AliasTy,
    pub term: TermKind,
}

#[derive(Clone, Debug)]
pub enum ImplPolarity {
    Positive,
    Negative,
    Reservation,
}

pub trait IndexedVal {
    fn to_val(index: usize) -> Self;

    fn to_index(&self) -> usize;
}
