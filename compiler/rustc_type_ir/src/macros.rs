/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
#[macro_export]
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl<I: $crate::Interner> $crate::TypeFoldable<I> for $ty {
                fn try_fold_with<F: $crate::FallibleTypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::TypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<I: $crate::Interner> $crate::TypeVisitable<I> for $ty {
                #[inline]
                fn visit_with<F: $crate::TypeVisitor<I>>(
                    &self,
                    _: &mut F)
                    -> F::Result
                {
                    <F::Result as $crate::VisitorResult>::output()
                }
            }

            // NOTE: this deliberately avoids adding an `I: Interner` generic, because that would
            // allow creating a trivial impl for arena-allocating types, which would be incorrect.
            unsafe impl<V> $crate::GenericTypeVisitable<V> for $ty {
                fn generic_visit_with(&self, _visitor: &mut V) {}
            }
        )+
    };
}

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeTraversalImpls! {
    (),
    bool,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    usize,
    // tidy-alphabetical-start
    crate::BoundConstness,
    crate::BoundVar,
    crate::ClausePolarity,
    crate::DebruijnIndex,
    crate::FloatTy,
    crate::InferConst,
    crate::InferTy,
    crate::IntTy,
    crate::RegionVid,
    crate::TypeFlags,
    crate::UintTy,
    crate::UniverseIndex,
    crate::Variance,
    crate::solve::BuiltinImplSource,
    crate::solve::Certainty,
    crate::solve::GoalSource,
    crate::solve::VisibleForLeakCheck,
    rustc_abi::ExternAbi,
    rustc_ast_ir::Mutability,
    // tidy-alphabetical-end
}

macro_rules! TrivialLiftImpls {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<I: $crate::Interner> $crate::lift::Lift<I> for $ty {
                type Lifted = Self;
                fn lift_to_interner(self, _: I) -> Self {
                    self
                }
            }
        )+
    };
}

TrivialLiftImpls! {
    crate::LateParamRegion<I>
}
