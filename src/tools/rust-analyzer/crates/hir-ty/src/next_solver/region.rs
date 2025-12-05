//! Things related to regions.

use hir_def::LifetimeParamId;
use intern::Symbol;
use rustc_type_ir::{
    BoundVar, BoundVarIndexKind, DebruijnIndex, Flags, INNERMOST, RegionVid, TypeFlags,
    TypeFoldable, TypeVisitable,
    inherent::{IntoKind, PlaceholderLike, SliceLike},
    relate::Relate,
};

use crate::next_solver::{GenericArg, OutlivesPredicate};

use super::{
    ErrorGuaranteed, SolverDefId, interned_vec_db,
    interner::{BoundVarKind, DbInterner, Placeholder},
};

pub type RegionKind<'db> = rustc_type_ir::RegionKind<DbInterner<'db>>;

#[salsa::interned(constructor = new_)]
pub struct Region<'db> {
    #[returns(ref)]
    kind_: RegionKind<'db>,
}

impl std::fmt::Debug for Region<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind().fmt(f)
    }
}

impl<'db> Region<'db> {
    pub fn new(interner: DbInterner<'db>, kind: RegionKind<'db>) -> Self {
        Region::new_(interner.db(), kind)
    }

    pub fn inner(&self) -> &RegionKind<'db> {
        crate::with_attached_db(|db| {
            let inner = self.kind_(db);
            // SAFETY: The caller already has access to a `Region<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute::<&RegionKind<'_>, &RegionKind<'db>>(inner) }
        })
    }

    pub fn new_early_param(
        interner: DbInterner<'db>,
        early_bound_region: EarlyParamRegion,
    ) -> Self {
        Region::new(interner, RegionKind::ReEarlyParam(early_bound_region))
    }

    pub fn new_placeholder(interner: DbInterner<'db>, placeholder: PlaceholderRegion) -> Self {
        Region::new(interner, RegionKind::RePlaceholder(placeholder))
    }

    pub fn new_var(interner: DbInterner<'db>, v: RegionVid) -> Region<'db> {
        Region::new(interner, RegionKind::ReVar(v))
    }

    pub fn new_erased(interner: DbInterner<'db>) -> Region<'db> {
        Region::new(interner, RegionKind::ReErased)
    }

    pub fn new_bound(
        interner: DbInterner<'db>,
        index: DebruijnIndex,
        bound: BoundRegion,
    ) -> Region<'db> {
        Region::new(interner, RegionKind::ReBound(BoundVarIndexKind::Bound(index), bound))
    }

    pub fn is_placeholder(&self) -> bool {
        matches!(self.inner(), RegionKind::RePlaceholder(..))
    }

    pub fn is_static(&self) -> bool {
        matches!(self.inner(), RegionKind::ReStatic)
    }

    pub fn is_erased(&self) -> bool {
        matches!(self.inner(), RegionKind::ReErased)
    }

    pub fn is_var(&self) -> bool {
        matches!(self.inner(), RegionKind::ReVar(_))
    }

    pub fn is_error(&self) -> bool {
        matches!(self.inner(), RegionKind::ReError(_))
    }

    pub fn error(interner: DbInterner<'db>) -> Self {
        Region::new(interner, RegionKind::ReError(ErrorGuaranteed))
    }

    pub fn type_flags(&self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match &self.inner() {
            RegionKind::ReVar(..) => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
                flags |= TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags |= TypeFlags::HAS_RE_INFER;
            }
            RegionKind::RePlaceholder(..) => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
                flags |= TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags |= TypeFlags::HAS_RE_PLACEHOLDER;
            }
            RegionKind::ReEarlyParam(..) => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
                flags |= TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags |= TypeFlags::HAS_RE_PARAM;
            }
            RegionKind::ReLateParam(..) => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
                flags |= TypeFlags::HAS_FREE_LOCAL_REGIONS;
            }
            RegionKind::ReStatic => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
            }
            RegionKind::ReBound(BoundVarIndexKind::Canonical, ..) => {
                flags |= TypeFlags::HAS_RE_BOUND;
                flags |= TypeFlags::HAS_CANONICAL_BOUND;
            }
            RegionKind::ReBound(BoundVarIndexKind::Bound(..), ..) => {
                flags |= TypeFlags::HAS_RE_BOUND;
            }
            RegionKind::ReErased => {
                flags |= TypeFlags::HAS_RE_ERASED;
            }
            RegionKind::ReError(..) => {
                flags |= TypeFlags::HAS_FREE_REGIONS;
                flags |= TypeFlags::HAS_ERROR;
            }
        }

        flags
    }
}

pub type PlaceholderRegion = Placeholder<BoundRegion>;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct EarlyParamRegion {
    // FIXME: See `ParamTy`.
    pub id: LifetimeParamId,
    pub index: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
/// The parameter representation of late-bound function parameters, "some region
/// at least as big as the scope `fr.scope`".
///
/// Similar to a placeholder region as we create `LateParam` regions when entering a binder
/// except they are always in the root universe and instead of using a boundvar to distinguish
/// between others we use the `DefId` of the parameter. For this reason the `bound_region` field
/// should basically always be `BoundRegionKind::Named` as otherwise there is no way of telling
/// different parameters apart.
pub struct LateParamRegion {
    pub scope: SolverDefId,
    pub bound_region: BoundRegionKind,
}

impl std::fmt::Debug for LateParamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReLateParam({:?}, {:?})", self.scope, self.bound_region)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoundRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    Anon,

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The `DefId` is needed to distinguish free regions in
    /// the event of shadowing.
    Named(SolverDefId),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

impl std::fmt::Debug for BoundRegionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            BoundRegionKind::Anon => write!(f, "BrAnon"),
            BoundRegionKind::Named(did) => {
                write!(f, "BrNamed({did:?})")
            }
            BoundRegionKind::ClosureEnv => write!(f, "BrEnv"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct BoundRegion {
    pub var: BoundVar,
    pub kind: BoundRegionKind,
}

impl rustc_type_ir::inherent::ParamLike for EarlyParamRegion {
    fn index(self) -> u32 {
        self.index
    }
}

impl std::fmt::Debug for EarlyParamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
        // write!(f, "{}/#{}", self.name, self.index)
    }
}

impl<'db> rustc_type_ir::inherent::BoundVarLike<DbInterner<'db>> for BoundRegion {
    fn var(self) -> BoundVar {
        self.var
    }

    fn assert_eq(self, var: BoundVarKind) {
        assert_eq!(self.kind, var.expect_region())
    }
}

impl core::fmt::Debug for BoundRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            BoundRegionKind::Anon => write!(f, "{:?}", self.var),
            BoundRegionKind::ClosureEnv => write!(f, "{:?}.Env", self.var),
            BoundRegionKind::Named(def) => {
                write!(f, "{:?}.Named({:?})", self.var, def)
            }
        }
    }
}

impl BoundRegionKind {
    pub fn is_named(&self) -> bool {
        matches!(self, BoundRegionKind::Named(_))
    }

    pub fn get_name(&self) -> Option<Symbol> {
        None
    }

    pub fn get_id(&self) -> Option<SolverDefId> {
        match self {
            BoundRegionKind::Named(id) => Some(*id),
            _ => None,
        }
    }
}

impl<'db> IntoKind for Region<'db> {
    type Kind = RegionKind<'db>;

    fn kind(self) -> Self::Kind {
        *self.inner()
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Region<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_region(*self)
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Region<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_region(self)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        folder.fold_region(self)
    }
}

impl<'db> Relate<DbInterner<'db>> for Region<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        relation.regions(a, b)
    }
}

impl<'db> Flags for Region<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        self.type_flags()
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        match &self.inner() {
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), _) => debruijn.shifted_in(1),
            _ => INNERMOST,
        }
    }
}

impl<'db> rustc_type_ir::inherent::Region<DbInterner<'db>> for Region<'db> {
    fn new_bound(
        interner: DbInterner<'db>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: BoundRegion,
    ) -> Self {
        Region::new(interner, RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), var))
    }

    fn new_anon_bound(
        interner: DbInterner<'db>,
        debruijn: rustc_type_ir::DebruijnIndex,
        var: rustc_type_ir::BoundVar,
    ) -> Self {
        Region::new(
            interner,
            RegionKind::ReBound(
                BoundVarIndexKind::Bound(debruijn),
                BoundRegion { var, kind: BoundRegionKind::Anon },
            ),
        )
    }

    fn new_canonical_bound(interner: DbInterner<'db>, var: rustc_type_ir::BoundVar) -> Self {
        Region::new(
            interner,
            RegionKind::ReBound(
                BoundVarIndexKind::Canonical,
                BoundRegion { var, kind: BoundRegionKind::Anon },
            ),
        )
    }

    fn new_static(interner: DbInterner<'db>) -> Self {
        Region::new(interner, RegionKind::ReStatic)
    }

    fn new_placeholder(
        interner: DbInterner<'db>,
        var: <DbInterner<'db> as rustc_type_ir::Interner>::PlaceholderRegion,
    ) -> Self {
        Region::new(interner, RegionKind::RePlaceholder(var))
    }
}

impl<'db> PlaceholderLike<DbInterner<'db>> for PlaceholderRegion {
    type Bound = BoundRegion;

    fn universe(self) -> rustc_type_ir::UniverseIndex {
        self.universe
    }

    fn var(self) -> rustc_type_ir::BoundVar {
        self.bound.var
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        Placeholder { universe: ui, bound: self.bound }
    }

    fn new(ui: rustc_type_ir::UniverseIndex, bound: Self::Bound) -> Self {
        Placeholder { universe: ui, bound }
    }

    fn new_anon(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        Placeholder { universe: ui, bound: BoundRegion { var, kind: BoundRegionKind::Anon } }
    }
}

type GenericArgOutlivesPredicate<'db> = OutlivesPredicate<'db, GenericArg<'db>>;

interned_vec_db!(RegionAssumptions, GenericArgOutlivesPredicate);
