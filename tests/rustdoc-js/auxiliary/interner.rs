#![feature(associated_type_defaults)]

use std::cmp::Ord;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::ControlFlow;

pub trait Interner: Sized {
    type DefId: Copy + Debug + Hash + Ord;
    type AdtDef: Copy + Debug + Hash + Ord;
    type GenericArgs: Copy
        + DebugWithInfcx<Self>
        + Hash
        + Ord
        + IntoIterator<Item = Self::GenericArg>;
    type GenericArg: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type Term: Copy + Debug + Hash + Ord;
    type Binder<T: TypeVisitable<Self>>: BoundVars<Self> + TypeSuperVisitable<Self>;
    type BoundVars: IntoIterator<Item = Self::BoundVar>;
    type BoundVar;
    type CanonicalVarKinds: Copy + Debug + Hash + Eq + IntoIterator<Item = CanonicalVarKind<Self>>;
    type Ty: Copy
        + DebugWithInfcx<Self>
        + Hash
        + Ord
        + Into<Self::GenericArg>
        + IntoKind<Kind = TyKind<Self>>
        + TypeSuperVisitable<Self>
        + Flags
        + Ty<Self>;
    type Tys: Copy + Debug + Hash + Ord + IntoIterator<Item = Self::Ty>;
    type AliasTy: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type ParamTy: Copy + Debug + Hash + Ord;
    type BoundTy: Copy + Debug + Hash + Ord;
    type PlaceholderTy: Copy + Debug + Hash + Ord + PlaceholderLike;
    type ErrorGuaranteed: Copy + Debug + Hash + Ord;
    type BoundExistentialPredicates: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type PolyFnSig: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type AllocId: Copy + Debug + Hash + Ord;
    type Const: Copy
        + DebugWithInfcx<Self>
        + Hash
        + Ord
        + Into<Self::GenericArg>
        + IntoKind<Kind = ConstKind<Self>>
        + ConstTy<Self>
        + TypeSuperVisitable<Self>
        + Flags
        + Const<Self>;
    type AliasConst: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type PlaceholderConst: Copy + Debug + Hash + Ord + PlaceholderLike;
    type ParamConst: Copy + Debug + Hash + Ord;
    type BoundConst: Copy + Debug + Hash + Ord;
    type ValueConst: Copy + Debug + Hash + Ord;
    type ExprConst: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type Region: Copy
        + DebugWithInfcx<Self>
        + Hash
        + Ord
        + Into<Self::GenericArg>
        + IntoKind<Kind = RegionKind<Self>>
        + Flags
        + Region<Self>;
    type EarlyParamRegion: Copy + Debug + Hash + Ord;
    type LateParamRegion: Copy + Debug + Hash + Ord;
    type BoundRegion: Copy + Debug + Hash + Ord;
    type InferRegion: Copy + DebugWithInfcx<Self> + Hash + Ord;
    type PlaceholderRegion: Copy + Debug + Hash + Ord + PlaceholderLike;
    type Predicate: Copy + Debug + Hash + Eq + TypeSuperVisitable<Self> + Flags;
    type TraitPredicate: Copy + Debug + Hash + Eq;
    type RegionOutlivesPredicate: Copy + Debug + Hash + Eq;
    type TypeOutlivesPredicate: Copy + Debug + Hash + Eq;
    type ProjectionPredicate: Copy + Debug + Hash + Eq;
    type NormalizesTo: Copy + Debug + Hash + Eq;
    type SubtypePredicate: Copy + Debug + Hash + Eq;
    type CoercePredicate: Copy + Debug + Hash + Eq;
    type ClosureKind: Copy + Debug + Hash + Eq;

    // Required method
    fn mk_canonical_var_kinds(self, kinds: &[CanonicalVarKind<Self>]) -> Self::CanonicalVarKinds;
}

pub trait DebugWithInfcx<I: Interner>: Debug {
    // Required method
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result;
}

pub trait TypeVisitable<I: Interner>: Debug + Clone {
    // Required method
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result;
}

pub trait BoundVars<I: Interner> {
    // Required methods
    fn bound_vars(&self) -> I::BoundVars;
    fn has_no_bound_vars(&self) -> bool;
}

pub trait TypeSuperVisitable<I: Interner>: TypeVisitable<I> {
    // Required method
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result;
}

pub struct CanonicalVarKind<I>(std::marker::PhantomData<I>);

pub struct TyKind<I>(std::marker::PhantomData<I>);

pub trait IntoKind {
    type Kind;

    // Required method
    fn kind(self) -> Self::Kind;
}
pub trait Flags {
    // Required methods
    fn flags(&self) -> TypeFlags;
    fn outer_exclusive_binder(&self) -> DebruijnIndex;
}
pub struct TypeFlags;

pub trait Ty<I: Interner<Ty = Self>> {
    // Required method
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait PlaceholderLike {
    // Required methods
    fn universe(self) -> UniverseIndex;
    fn var(self) -> BoundVar;
    fn with_updated_universe(self, ui: UniverseIndex) -> Self;
    fn new(ui: UniverseIndex, var: BoundVar) -> Self;
}

pub struct UniverseIndex;

pub struct BoundVar;

pub struct ConstKind<I>(std::marker::PhantomData<I>);
pub trait Const<I: Interner<Const = Self>> {
    // Required method
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar, ty: I::Ty) -> Self;
}

pub trait ConstTy<I: Interner> {
    // Required method
    fn ty(self) -> I::Ty;
}

pub struct DebruijnIndex;

pub struct RegionKind<I>(std::marker::PhantomData<I>);
pub trait Region<I: Interner<Region = Self>> {
    // Required method
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait TypeVisitor<I: Interner>: Sized {
    type Result: VisitorResult = ();

    // Provided methods
    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &I::Binder<T>) -> Self::Result {
        unimplemented!()
    }
    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        unimplemented!()
    }
    fn visit_region(&mut self, _r: I::Region) -> Self::Result {
        unimplemented!()
    }
    fn visit_const(&mut self, c: I::Const) -> Self::Result {
        unimplemented!()
    }
    fn visit_predicate(&mut self, p: I::Predicate) -> Self::Result {
        unimplemented!()
    }
}

pub trait VisitorResult {
    type Residual;

    // Required methods
    fn output() -> Self;
    fn from_residual(residual: Self::Residual) -> Self;
    fn from_branch(b: ControlFlow<Self::Residual>) -> Self;
    fn branch(self) -> ControlFlow<Self::Residual>;
}

impl VisitorResult for () {
    type Residual = ();
    fn output() -> Self {}
    fn from_residual(_: Self::Residual) -> Self {}
    fn from_branch(_: ControlFlow<Self::Residual>) -> Self {}
    fn branch(self) -> ControlFlow<Self::Residual> {
        ControlFlow::Continue(())
    }
}

pub struct WithInfcx<'a, Infcx: InferCtxtLike, T> {
    pub data: T,
    pub infcx: &'a Infcx,
}

pub trait InferCtxtLike {
    type Interner: Interner;

    // Required methods
    fn interner(&self) -> Self::Interner;
    fn universe_of_ty(&self, ty: TyVid) -> Option<UniverseIndex>;
    fn root_ty_var(&self, vid: TyVid) -> TyVid;
    fn probe_ty_var(&self, vid: TyVid) -> Option<<Self::Interner as Interner>::Ty>;
    fn universe_of_lt(
        &self,
        lt: <Self::Interner as Interner>::InferRegion,
    ) -> Option<UniverseIndex>;
    fn opportunistic_resolve_lt_var(
        &self,
        vid: <Self::Interner as Interner>::InferRegion,
    ) -> Option<<Self::Interner as Interner>::Region>;
    fn universe_of_ct(&self, ct: ConstVid) -> Option<UniverseIndex>;
    fn root_ct_var(&self, vid: ConstVid) -> ConstVid;
    fn probe_ct_var(&self, vid: ConstVid) -> Option<<Self::Interner as Interner>::Const>;
}

pub struct TyVid;
pub struct ConstVid;
