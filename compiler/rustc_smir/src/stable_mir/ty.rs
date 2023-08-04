use rustc_middle::mir::interpret::{alloc_range, AllocRange, ConstValue, Pointer};

use super::{mir::Mutability, mir::Safety, with, DefId};
use crate::{
    rustc_internal::{opaque, Opaque},
    rustc_smir::{Stable, Tables},
};

#[derive(Copy, Clone, Debug)]
pub struct Ty(pub usize);

impl Ty {
    pub fn kind(&self) -> TyKind {
        with(|context| context.ty_kind(*self))
    }
}

#[derive(Debug, Clone)]
pub struct Const {
    pub literal: ConstantKind,
}

type Ident = Opaque;
pub(crate) type Region = Opaque;
pub(crate) type Span = Opaque;

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
    Generator(GeneratorDef, GenericArgs, Movability),
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
    F16,
    F32,
    F64,
    F128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ForeignDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FnDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClosureDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GeneratorDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BrNamedDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AdtDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AliasDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TraitDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GenericDef(pub(crate) DefId);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConstDef(pub(crate) DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ImplDef(pub(crate) DefId);

#[derive(Clone, Debug)]
pub struct GenericArgs(pub Vec<GenericArgKind>);

#[derive(Clone, Debug)]
pub enum GenericArgKind {
    Lifetime(Region),
    Type(Ty),
    Const(Const),
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
    BrAnon(Option<Span>),
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
pub type Prov = Opaque;
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

impl Allocation {
    /// Creates new empty `Allocation` from given `Align`.
    fn new_empty_allocation(align: rustc_target::abi::Align) -> Allocation {
        Allocation {
            bytes: Vec::new(),
            provenance: ProvenanceMap { ptrs: Vec::new() },
            align: align.bytes(),
            mutability: Mutability::Not,
        }
    }
}

// We need this method instead of a Stable implementation
// because we need to get `Ty` of the const we are trying to create, to do that
// we need to have access to `ConstantKind` but we can't access that inside Stable impl.
pub fn new_allocation<'tcx>(
    const_kind: &rustc_middle::mir::ConstantKind<'tcx>,
    const_value: ConstValue<'tcx>,
    tables: &mut Tables<'tcx>,
) -> Allocation {
    match const_value {
        ConstValue::Scalar(scalar) => {
            let size = scalar.size();
            let align = tables
                .tcx
                .layout_of(rustc_middle::ty::ParamEnv::reveal_all().and(const_kind.ty()))
                .unwrap()
                .align;
            let mut allocation = rustc_middle::mir::interpret::Allocation::uninit(size, align.abi);
            allocation
                .write_scalar(&tables.tcx, alloc_range(rustc_target::abi::Size::ZERO, size), scalar)
                .unwrap();
            allocation.stable(tables)
        }
        ConstValue::ZeroSized => {
            let align = tables
                .tcx
                .layout_of(rustc_middle::ty::ParamEnv::empty().and(const_kind.ty()))
                .unwrap()
                .align;
            Allocation::new_empty_allocation(align.abi)
        }
        ConstValue::Slice { data, start, end } => {
            let alloc_id = tables.tcx.create_memory_alloc(data);
            let ptr = Pointer::new(alloc_id, rustc_target::abi::Size::from_bytes(start));
            let scalar_ptr = rustc_middle::mir::interpret::Scalar::from_pointer(ptr, &tables.tcx);
            let scalar_len = rustc_middle::mir::interpret::Scalar::from_target_usize(
                (end - start) as u64,
                &tables.tcx,
            );
            let layout = tables
                .tcx
                .layout_of(rustc_middle::ty::ParamEnv::reveal_all().and(const_kind.ty()))
                .unwrap();
            let mut allocation =
                rustc_middle::mir::interpret::Allocation::uninit(layout.size, layout.align.abi);
            allocation
                .write_scalar(
                    &tables.tcx,
                    alloc_range(rustc_target::abi::Size::ZERO, tables.tcx.data_layout.pointer_size),
                    scalar_ptr,
                )
                .unwrap();
            allocation
                .write_scalar(
                    &tables.tcx,
                    alloc_range(tables.tcx.data_layout.pointer_size, scalar_len.size()),
                    scalar_len,
                )
                .unwrap();
            allocation.stable(tables)
        }
        ConstValue::ByRef { alloc, offset } => {
            let ty_size = tables
                .tcx
                .layout_of(rustc_middle::ty::ParamEnv::reveal_all().and(const_kind.ty()))
                .unwrap()
                .size;
            allocation_filter(&alloc.0, alloc_range(offset, ty_size), tables)
        }
    }
}

/// Creates an `Allocation` only from information within the `AllocRange`.
pub fn allocation_filter<'tcx>(
    alloc: &rustc_middle::mir::interpret::Allocation,
    alloc_range: AllocRange,
    tables: &mut Tables<'tcx>,
) -> Allocation {
    let mut bytes: Vec<Option<u8>> = alloc
        .inspect_with_uninit_and_ptr_outside_interpreter(
            alloc_range.start.bytes_usize()..alloc_range.end().bytes_usize(),
        )
        .iter()
        .copied()
        .map(Some)
        .collect();
    for (i, b) in bytes.iter_mut().enumerate() {
        if !alloc
            .init_mask()
            .get(rustc_target::abi::Size::from_bytes(i + alloc_range.start.bytes_usize()))
        {
            *b = None;
        }
    }
    let mut ptrs = Vec::new();
    for (offset, prov) in alloc
        .provenance()
        .ptrs()
        .iter()
        .filter(|a| a.0 >= alloc_range.start && a.0 <= alloc_range.end())
    {
        ptrs.push((offset.bytes_usize() - alloc_range.start.bytes_usize(), opaque(prov)));
    }
    Allocation {
        bytes: bytes,
        provenance: ProvenanceMap { ptrs },
        align: alloc.align.bytes(),
        mutability: alloc.mutability.stable(tables),
    }
}

#[derive(Clone, Debug)]
pub enum ConstantKind {
    Allocated(Allocation),
    Unevaluated(UnevaluatedConst),
    ParamCt(Opaque),
}

#[derive(Clone, Debug)]
pub struct UnevaluatedConst {
    pub ty: Ty,
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
