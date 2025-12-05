/// Helper trait that allows `arena_cache` queries to return `Option<&T>`
/// instead of `&Option<T>`, and avoid allocating `None` in the arena.
///
/// An arena-cached query must be declared to return a type that implements
/// this trait, i.e. either `&'tcx T` or `Option<&'tcx T>`. This trait then
/// determines the types returned by the provider and stored in the arena,
/// and provides a function to bridge between the three types.
pub trait ArenaCached<'tcx>: Sized {
    /// Type that is returned by the query provider.
    type Provided;
    /// Type that is stored in the arena.
    type Allocated: 'tcx;

    /// Takes a provided value, and allocates it in the arena (if appropriate)
    /// with the help of the given `arena_alloc` closure.
    fn alloc_in_arena(
        arena_alloc: impl Fn(Self::Allocated) -> &'tcx Self::Allocated,
        value: Self::Provided,
    ) -> Self;
}

impl<'tcx, T> ArenaCached<'tcx> for &'tcx T {
    type Provided = T;
    type Allocated = T;

    fn alloc_in_arena(
        arena_alloc: impl Fn(Self::Allocated) -> &'tcx Self::Allocated,
        value: Self::Provided,
    ) -> Self {
        // Just allocate in the arena normally.
        arena_alloc(value)
    }
}

impl<'tcx, T> ArenaCached<'tcx> for Option<&'tcx T> {
    type Provided = Option<T>;
    /// The provide value is `Option<T>`, but we only store `T` in the arena.
    type Allocated = T;

    fn alloc_in_arena(
        arena_alloc: impl Fn(Self::Allocated) -> &'tcx Self::Allocated,
        value: Self::Provided,
    ) -> Self {
        // Don't store None in the arena, and wrap the allocated reference in Some.
        value.map(arena_alloc)
    }
}
