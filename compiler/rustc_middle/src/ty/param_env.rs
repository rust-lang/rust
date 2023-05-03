use std::fmt;
use std::ops::ControlFlow;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::tagged_ptr::CopyTaggedPtr;
use rustc_hir as hir;
use rustc_query_system::ich::StableHashingContext;

use crate::traits;
use crate::ty::{
    self, list::List, Predicate, TyCtxt, TypeFoldable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};

/// When type checking, we use the `ParamEnv` to track
/// details about the set of where-clauses that are in scope at this
/// particular point.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ParamEnv<'tcx> {
    /// This packs both caller bounds and the reveal enum into one pointer.
    ///
    /// Caller bounds are `Obligation`s that the caller must satisfy. This is
    /// basically the set of bounds on the in-scope type parameters, translated
    /// into `Obligation`s, and elaborated and normalized.
    ///
    /// Use the `caller_bounds()` method to access.
    ///
    /// Typically, this is `Reveal::UserFacing`, but during codegen we
    /// want `Reveal::All`.
    ///
    /// Note: This is packed, use the reveal() method to access it.
    packed: CopyTaggedPtr<&'tcx List<Predicate<'tcx>>, ParamTag, true>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TypeFoldable, TypeVisitable)]
#[derive(HashStable, Lift)]
pub struct ParamEnvAnd<'tcx, T> {
    pub param_env: ParamEnv<'tcx>,
    pub value: T,
}

#[derive(Copy, Clone)]
struct ParamTag {
    reveal: traits::Reveal,
    constness: hir::Constness,
}

impl<'tcx> ParamEnv<'tcx> {
    /// Construct a trait environment suitable for contexts where
    /// there are no where-clauses in scope. Hidden types (like `impl
    /// Trait`) are left hidden, so this is suitable for ordinary
    /// type-checking.
    #[inline]
    pub fn empty() -> Self {
        Self::new(List::empty(), traits::Reveal::UserFacing, hir::Constness::NotConst)
    }

    #[inline]
    pub fn caller_bounds(self) -> &'tcx List<Predicate<'tcx>> {
        self.packed.pointer()
    }

    #[inline]
    pub fn reveal(self) -> traits::Reveal {
        self.packed.tag().reveal
    }

    #[inline]
    pub fn constness(self) -> hir::Constness {
        self.packed.tag().constness
    }

    #[inline]
    pub fn is_const(self) -> bool {
        self.packed.tag().constness == hir::Constness::Const
    }

    /// Construct a trait environment with no where-clauses in scope
    /// where the values of all `impl Trait` and other hidden types
    /// are revealed. This is suitable for monomorphized, post-typeck
    /// environments like codegen or doing optimizations.
    ///
    /// N.B., if you want to have predicates in scope, use `ParamEnv::new`,
    /// or invoke `param_env.with_reveal_all()`.
    #[inline]
    pub fn reveal_all() -> Self {
        Self::new(List::empty(), traits::Reveal::All, hir::Constness::NotConst)
    }

    /// Construct a trait environment with the given set of predicates.
    #[inline]
    pub fn new(
        caller_bounds: &'tcx List<Predicate<'tcx>>,
        reveal: traits::Reveal,
        constness: hir::Constness,
    ) -> Self {
        ParamEnv { packed: CopyTaggedPtr::new(caller_bounds, ParamTag { reveal, constness }) }
    }

    pub fn with_user_facing(mut self) -> Self {
        self.packed.set_tag(ParamTag { reveal: traits::Reveal::UserFacing, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn with_constness(mut self, constness: hir::Constness) -> Self {
        self.packed.set_tag(ParamTag { constness, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn with_const(mut self) -> Self {
        self.packed.set_tag(ParamTag { constness: hir::Constness::Const, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn without_const(mut self) -> Self {
        self.packed.set_tag(ParamTag { constness: hir::Constness::NotConst, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn remap_constness_with(&mut self, mut constness: ty::BoundConstness) {
        *self = self.with_constness(constness.and(self.constness()))
    }

    /// Returns a new parameter environment with the same clauses, but
    /// which "reveals" the true results of projections in all cases
    /// (even for associated types that are specializable). This is
    /// the desired behavior during codegen and certain other special
    /// contexts; normally though we want to use `Reveal::UserFacing`,
    /// which is the default.
    /// All opaque types in the caller_bounds of the `ParamEnv`
    /// will be normalized to their underlying types.
    /// See PR #65989 and issue #65918 for more details
    pub fn with_reveal_all_normalized(self, tcx: TyCtxt<'tcx>) -> Self {
        if self.packed.tag().reveal == traits::Reveal::All {
            return self;
        }

        ParamEnv::new(
            tcx.reveal_opaque_types_in_bounds(self.caller_bounds()),
            traits::Reveal::All,
            self.constness(),
        )
    }

    /// Returns this same environment but with no caller bounds.
    #[inline]
    pub fn without_caller_bounds(self) -> Self {
        Self::new(List::empty(), self.reveal(), self.constness())
    }

    /// Creates a suitable environment in which to perform trait
    /// queries on the given value. When type-checking, this is simply
    /// the pair of the environment plus value. But when reveal is set to
    /// All, then if `value` does not reference any type parameters, we will
    /// pair it with the empty environment. This improves caching and is generally
    /// invisible.
    ///
    /// N.B., we preserve the environment when type-checking because it
    /// is possible for the user to have wacky where-clauses like
    /// `where Box<u32>: Copy`, which are clearly never
    /// satisfiable. We generally want to behave as if they were true,
    /// although the surrounding function is never reachable.
    pub fn and<T: TypeVisitable<TyCtxt<'tcx>>>(self, value: T) -> ParamEnvAnd<'tcx, T> {
        match self.reveal() {
            traits::Reveal::UserFacing => ParamEnvAnd { param_env: self, value },

            traits::Reveal::All => {
                if value.is_global() {
                    ParamEnvAnd { param_env: self.without_caller_bounds(), value }
                } else {
                    ParamEnvAnd { param_env: self, value }
                }
            }
        }
    }
}

impl<'tcx, T> ParamEnvAnd<'tcx, T> {
    pub fn into_parts(self) -> (ParamEnv<'tcx>, T) {
        (self.param_env, self.value)
    }
}

impl<'tcx> fmt::Debug for ParamEnv<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamEnv")
            .field("caller_bounds", &self.caller_bounds())
            .field("reveal", &self.reveal())
            .field("constness", &self.constness())
            .finish()
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ParamEnv<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.caller_bounds().hash_stable(hcx, hasher);
        self.reveal().hash_stable(hcx, hasher);
        self.constness().hash_stable(hcx, hasher);
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ParamEnv<'tcx> {
    fn try_fold_with<F: ty::fold::FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ParamEnv::new(
            self.caller_bounds().try_fold_with(folder)?,
            self.reveal().try_fold_with(folder)?,
            self.constness(),
        ))
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ParamEnv<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.caller_bounds().visit_with(visitor)?;
        self.reveal().visit_with(visitor)
    }
}

impl_tag! {
    impl Tag for ParamTag;
    ParamTag { reveal: traits::Reveal::UserFacing, constness: hir::Constness::NotConst },
    ParamTag { reveal: traits::Reveal::All,        constness: hir::Constness::NotConst },
    ParamTag { reveal: traits::Reveal::UserFacing, constness: hir::Constness::Const    },
    ParamTag { reveal: traits::Reveal::All,        constness: hir::Constness::Const    },
}
