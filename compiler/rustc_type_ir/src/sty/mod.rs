use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::StableHash_NoContext;
use rustc_type_ir_macros::{GenericTypeVisitable, Lift_Generic};
use tracing::debug;

use crate::inherent::*;
use crate::intern::Interned;
use crate::relate::{Relate, RelateResult, TypeRelation};
use crate::{
    BoundRegion, BoundRegionKind, BoundVar, BoundVarIndexKind, DebruijnIndex, FallibleTypeFolder,
    Flags, Interner, PlaceholderRegion, RegionKind, RegionVid, TypeFlags, TypeFoldable, TypeFolder,
    TypeVisitable, TypeVisitor,
};

/// Use this rather than `RegionKind`, whenever possible.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
#[derive(GenericTypeVisitable, Lift_Generic)]
pub struct Region<I: Interner>(pub I::InternedRegionKind);

// These are only the `inherent` trait methods that have been ported across
impl<I: Interner> Region<I> {
    #[inline]
    pub fn new_bound(interner: I, debruijn: DebruijnIndex, bound_region: BoundRegion<I>) -> Self {
        interner.intern_bound_region(debruijn, bound_region)
    }

    #[inline]
    pub fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Self::new_bound(interner, debruijn, BoundRegion { var, kind: BoundRegionKind::Anon })
    }

    #[inline]
    pub fn new_canonical_bound(interner: I, var: BoundVar) -> Self {
        interner.intern_canonical_bound(var)
    }

    #[inline]
    pub fn new_placeholder(interner: I, placeholder: PlaceholderRegion<I>) -> Self {
        interner.intern_region(RegionKind::RePlaceholder(placeholder))
    }

    #[inline]
    pub fn new_static(interner: I) -> Self {
        interner.get_re_static_lifetime()
    }

    #[inline]
    pub fn is_bound(self) -> bool {
        matches!(self.0.get(), RegionKind::ReBound(..))
    }

    #[inline]
    pub fn is_error(self) -> bool {
        matches!(self.kind(), RegionKind::ReError(_))
    }

    #[inline]
    pub fn is_static(self) -> bool {
        matches!(self.kind(), RegionKind::ReStatic)
    }

    #[inline]
    pub fn is_erased(self) -> bool {
        matches!(self.kind(), RegionKind::ReErased)
    }

    #[inline]
    pub fn is_placeholder(self) -> bool {
        matches!(self.kind(), RegionKind::RePlaceholder(..))
    }

    /// True for free regions other than `'static`.
    pub fn is_param(self) -> bool {
        matches!(self.kind(), RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_))
    }

    /// True for free region in the current context.
    ///
    /// This is the case for `'static` and param regions.
    pub fn is_free(self) -> bool {
        match self.kind() {
            RegionKind::ReStatic | RegionKind::ReEarlyParam(..) | RegionKind::ReLateParam(..) => {
                true
            }
            RegionKind::ReVar(..)
            | RegionKind::RePlaceholder(..)
            | RegionKind::ReBound(..)
            | RegionKind::ReErased
            | RegionKind::ReError(..) => false,
        }
    }

    pub fn is_var(self) -> bool {
        matches!(self.kind(), RegionKind::ReVar(_))
    }

    pub fn as_var(self) -> RegionVid {
        match self.kind() {
            RegionKind::ReVar(vid) => vid,
            _ => panic!("expected region {:?} to be of kind ReVar", self),
        }
    }

    // FIXME this should be made private and instead accessed via the
    // trait Flags
    #[inline]
    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match self.0.get() {
            RegionKind::ReVar(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
            }
            RegionKind::RePlaceholder(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PLACEHOLDER;
            }
            RegionKind::ReEarlyParam(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PARAM;
            }
            RegionKind::ReLateParam { .. } => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
            }
            RegionKind::ReStatic => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
            RegionKind::ReBound(BoundVarIndexKind::Canonical, _) => {
                flags = flags | TypeFlags::HAS_RE_BOUND;
                flags = flags | TypeFlags::HAS_CANONICAL_BOUND;
            }
            RegionKind::ReBound(BoundVarIndexKind::Bound(..), _) => {
                flags = flags | TypeFlags::HAS_RE_BOUND;
            }
            RegionKind::ReErased => {
                flags = flags | TypeFlags::HAS_RE_ERASED;
            }
            RegionKind::ReError(_) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_RE_ERROR;
            }
        }
        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    #[inline]
    pub fn kind(self) -> RegionKind<I> {
        self.0.get()
    }
}

impl<I: Interner> Flags for Region<I> {
    fn flags(&self) -> TypeFlags {
        self.type_flags()
    }

    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        match self.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), _) => debruijn.shifted_in(1),
            _ => crate::INNERMOST,
        }
    }
}

impl<I: Interner> IntoKind for Region<I> {
    type Kind = RegionKind<I>;

    fn kind(self) -> Self::Kind {
        self.0.get()
    }
}

impl<I: Interner> Relate<I> for Region<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: Region<I>,
        b: Region<I>,
    ) -> RelateResult<I, Region<I>> {
        relation.regions(a, b)
    }
}

impl<I: Interner> fmt::Debug for Region<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

impl<I: Interner> TypeVisitable<I> for Region<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_region(*self)
    }
}

impl<I: Interner> TypeFoldable<I> for Region<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_region(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_region(self)
    }
}
