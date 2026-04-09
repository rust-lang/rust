use std::fmt;

use derive_where::derive_where;
use rustc_data_structures::intern::Interned;
#[cfg(feature = "nightly")]
use rustc_macros::HashStable_NoContext;
#[cfg(feature = "nightly")]
use rustc_serialize::{Decodable, Encodable};
use tracing::debug;

use crate::inherent::*;
use crate::relate::{Relate, RelateResult, TypeRelation};
use crate::{
    BoundRegion, BoundRegionKind, BoundVar, BoundVarIndexKind, DebruijnIndex, FallibleTypeFolder,
    Flags, Interner, PlaceholderRegion, RegionKind, TypeFlags, TypeFoldable, TypeFolder,
    TypeVisitable, TypeVisitor,
};

/// Use this rather than `RegionKind`, whenever possible.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
#[rustc_pass_by_value]
pub struct Region<I: Interner>(pub I::InternedRegionKind);

// These are only the `inherent` trait methods that have been ported across
impl<I: Interner> Region<I> {
    #[inline]
    pub fn new_bound(interner: I, debruijn: DebruijnIndex, bound_region: BoundRegion<I>) -> Self {
        // Use a pre-interned one when possible.
        if let BoundRegion { var, kind: BoundRegionKind::Anon } = bound_region
            && let Some(re) =
                interner.get_anon_re_bounds_lifetime(debruijn.as_usize(), var.as_usize())
        {
            re
        } else {
            interner.intern_region(RegionKind::ReBound(
                BoundVarIndexKind::Bound(debruijn),
                bound_region,
            ))
        }
    }

    #[inline]
    pub fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Self::new_bound(interner, debruijn, BoundRegion { var, kind: BoundRegionKind::Anon })
    }

    #[inline]
    pub fn new_canonical_bound(interner: I, var: BoundVar) -> Self {
        // Use a pre-interned one when possible.
        if let Some(re) = interner.get_anon_re_canonical_bounds_lifetime(var.as_usize()) {
            re
        } else {
            interner.intern_region(RegionKind::ReBound(
                BoundVarIndexKind::Canonical,
                BoundRegion { var, kind: BoundRegionKind::Anon },
            ))
        }
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
        matches!(self.0.kind(), RegionKind::ReBound(..))
    }

    #[inline]
    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match self.0.kind() {
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
        self.0.kind()
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
        self.0.kind()
    }
}

impl<'tcx, T: Copy> IntoKind for Interned<'tcx, T> {
    type Kind = T;

    fn kind(self) -> Self::Kind {
        *self.0
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

#[cfg(feature = "nightly")]
impl<I: Interner, E: rustc_serialize::Encoder> Encodable<E> for Region<I>
where
    RegionKind<I>: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        self.kind().encode(e);
    }
}

#[cfg(feature = "nightly")]
impl<I: Interner, D: crate::InternerDecoder<Interner = I>> Decodable<D> for Region<I>
where
    RegionKind<I>: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        decoder.interner().intern_region(Decodable::decode(decoder))
    }
}
