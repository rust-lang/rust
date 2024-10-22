/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl<I: $crate::Interner> $crate::traverse::TypeTraversable<I> for $ty {
                type Kind = $crate::traverse::NoopTypeTraversal;
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
