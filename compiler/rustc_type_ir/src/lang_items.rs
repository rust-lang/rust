/// Lang items used by the new trait solver. This can be mapped to whatever internal
/// representation of `LangItem`s used in the underlying compiler implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverProjectionLangItem {
    // tidy-alphabetical-start
    AsyncFnKindUpvars,
    AsyncFnOnceOutput,
    CallOnceFuture,
    CallRefFuture,
    CoroutineReturn,
    CoroutineYield,
    FieldBase,
    FieldType,
    FutureOutput,
    Metadata,
    // tidy-alphabetical-end
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverAdtLangItem {
    // tidy-alphabetical-start
    DynMetadata,
    Option,
    Poll,
    // tidy-alphabetical-end
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverTraitLangItem {
    // tidy-alphabetical-start
    AsyncFn,
    AsyncFnKindHelper,
    AsyncFnMut,
    AsyncFnOnce,
    AsyncIterator,
    BikeshedGuaranteedNoDrop,
    Clone,
    Copy,
    Coroutine,
    Destruct,
    DiscriminantKind,
    Drop,
    Field,
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
    TrivialClone,
    Tuple,
    Unpin,
    Unsize,
    // tidy-alphabetical-end
}
