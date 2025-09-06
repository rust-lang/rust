/// Lang items used by the new trait solver. This can be mapped to whatever internal
/// representation of `LangItem`s used in the underlying compiler implementation.
pub enum SolverLangItem {
    // tidy-alphabetical-start
    AsyncFnKindUpvars,
    AsyncFnOnceOutput,
    CallOnceFuture,
    CallRefFuture,
    CoroutineReturn,
    CoroutineYield,
    DynMetadata,
    FutureOutput,
    Metadata,
    // tidy-alphabetical-end
}

pub enum SolverAdtLangItem {
    // tidy-alphabetical-start
    Option,
    Poll,
    // tidy-alphabetical-end
}

pub enum SolverTraitLangItem {
    // tidy-alphabetical-start
    AsyncFn,
    AsyncFnKindHelper,
    AsyncFnMut,
    AsyncFnOnce,
    AsyncFnOnceOutput,
    AsyncIterator,
    BikeshedGuaranteedNoDrop,
    Clone,
    Copy,
    Coroutine,
    Destruct,
    DiscriminantKind,
    Drop,
    Fn,
    FnMut,
    FnOnce,
    FnPtrTrait,
    FusedIterator,
    Future,
    Iterator,
    MetaSized,
    PointeeSized,
    PointeeTrait,
    Sized,
    TransmuteTrait,
    Tuple,
    Unpin,
    Unsize,
    // tidy-alphabetical-end
}
