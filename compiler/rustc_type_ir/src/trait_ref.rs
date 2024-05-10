use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::inherent::*;
use crate::lift::Lift;
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::Interner;

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where-clause:
/// ```ignore (illustrative)
/// T: Foo<U>
/// ```
/// This would be represented by a trait-reference where the `DefId` is the
/// `DefId` for the trait `Foo` and the args define `T` as parameter 0,
/// and `U` as parameter 1.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the generic parameters.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct TraitRef<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    /// This field exists to prevent the creation of `TraitRef` without
    /// calling [`TraitRef::new`].
    _use_trait_ref_new_instead: (),
}

impl<I: Interner> TraitRef<I> {
    pub fn new(
        interner: I,
        trait_def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        let args = interner.check_and_mk_args(trait_def_id, args);
        Self { def_id: trait_def_id, args, _use_trait_ref_new_instead: () }
    }

    pub fn from_method(interner: I, trait_id: I::DefId, args: I::GenericArgs) -> TraitRef<I> {
        let generics = interner.generics_of(trait_id);
        TraitRef::new(interner, trait_id, args.into_iter().take(generics.count()))
    }

    /// Returns a `TraitRef` of the form `P0: Foo<P1..Pn>` where `Pi`
    /// are the parameters defined on trait.
    pub fn identity(interner: I, def_id: I::DefId) -> TraitRef<I> {
        TraitRef::new(interner, def_id, I::GenericArgs::identity_for_item(interner, def_id))
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        TraitRef::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.into_iter().skip(1)),
        )
    }

    #[inline]
    pub fn self_ty(&self) -> I::Ty {
        self.args.type_at(0)
    }
}

// FIXME(compiler-errors): Make this into a `Lift_Generic` impl.
impl<I: Interner, U: Interner> Lift<U> for TraitRef<I>
where
    I::DefId: Lift<U, Lifted = U::DefId>,
    I::GenericArgs: Lift<U, Lifted = U::GenericArgs>,
{
    type Lifted = TraitRef<U>;

    fn lift_to_tcx(self, tcx: U) -> Option<Self::Lifted> {
        Some(TraitRef {
            def_id: self.def_id.lift_to_tcx(tcx)?,
            args: self.args.lift_to_tcx(tcx)?,
            _use_trait_ref_new_instead: (),
        })
    }
}

impl<I: Interner> TypeVisitable<I> for TraitRef<I>
where
    I::GenericArgs: TypeVisitable<I>,
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        self.args.visit_with(visitor)
    }
}

impl<I: Interner> TypeFoldable<I> for TraitRef<I>
where
    I::GenericArgs: TypeFoldable<I>,
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(TraitRef {
            def_id: self.def_id,
            args: self.args.try_fold_with(folder)?,
            _use_trait_ref_new_instead: (),
        })
    }
}
