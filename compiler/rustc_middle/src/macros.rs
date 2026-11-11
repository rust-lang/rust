///////////////////////////////////////////////////////////////////////////
// Lift and TypeFoldable/TypeVisitable macros
//
// When possible, use one of these (relatively) convenient macros to write
// the impls for you.

macro_rules! TrivialLiftImpls {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'tcx> $crate::ty::Lift<$crate::ty::TyCtxt<'tcx>> for $ty {
                type Lifted = Self;
                fn lift_to_interner(self, _: $crate::ty::TyCtxt<'tcx>) -> Self {
                    self
                }
            }
        )+
    };
}

/// Used for types that are `Copy` and which **do not care about arena
/// allocated data** (i.e., don't need to be folded).
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'tcx> $crate::ty::TypeFoldable<$crate::ty::TyCtxt<'tcx>> for $ty {
                fn try_fold_with<F: $crate::ty::FallibleTypeFolder<$crate::ty::TyCtxt<'tcx>>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::ty::TypeFolder<$crate::ty::TyCtxt<'tcx>>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<'tcx> $crate::ty::TypeVisitable<$crate::ty::TyCtxt<'tcx>> for $ty {
                #[inline]
                fn visit_with<F: $crate::ty::TypeVisitor<$crate::ty::TyCtxt<'tcx>>>(
                    &self,
                    _: &mut F)
                    -> F::Result
                {
                    <F::Result as ::rustc_middle::ty::VisitorResult>::output()
                }
            }
        )+
    };
}

macro_rules! TrivialTypeTraversalAndLiftImpls {
    ($($t:tt)*) => {
        TrivialTypeTraversalImpls! { $($t)* }
        TrivialLiftImpls! { $($t)* }
    }
}
