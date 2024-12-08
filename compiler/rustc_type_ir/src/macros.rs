/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl<I: $crate::Interner> $crate::fold::TypeFoldable<I> for $ty {
                fn try_fold_with<F: $crate::fold::FallibleTypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::fold::TypeFolder<I>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<I: $crate::Interner> $crate::visit::TypeVisitable<I> for $ty {
                #[inline]
                fn visit_with<F: $crate::visit::TypeVisitor<I>>(
                    &self,
                    _: &mut F)
                    -> F::Result
                {
                    <F::Result as rustc_ast_ir::visit::VisitorResult>::output()
                }
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
    usize,
    u16,
    u32,
    u64,
    String,
    crate::AliasRelationDirection,
    crate::AliasTyKind,
    crate::BoundConstness,
    crate::DebruijnIndex,
    crate::FloatTy,
    crate::InferTy,
    crate::IntVarValue,
    crate::PredicatePolarity,
    crate::RegionVid,
    crate::solve::BuiltinImplSource,
    crate::solve::Certainty,
    crate::solve::GoalSource,
    crate::solve::MaybeCause,
    crate::solve::NoSolution,
    crate::UniverseIndex,
    crate::Variance,
    rustc_ast_ir::Movability,
    rustc_ast_ir::Mutability,
}
