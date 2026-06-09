/// An `Into`-like trait that takes `TyCtxt` to perform interner-specific transformations.
pub trait Upcast<I, T> {
    fn upcast(self, interner: I) -> T;
}

impl<I, T, U> Upcast<I, U> for T
where
    U: UpcastFrom<I, T>,
{
    fn upcast(self, interner: I) -> U {
        U::upcast_from(self, interner)
    }
}

/// A `From`-like trait that takes `TyCtxt` to perform interner-specific transformations.
pub trait UpcastFrom<I, T> {
    fn upcast_from(from: T, interner: I) -> Self;
}

impl<I, T> UpcastFrom<I, T> for T {
    fn upcast_from(from: T, _tcx: I) -> Self {
        from
    }
}
