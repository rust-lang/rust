use std::mem;

use rustc_arena::TypedArena;

use crate::ty::TyCtxt;

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

    /// Takes a provided value, and allocates it in an appropriate arena,
    /// unless the particular value doesn't need allocation (e.g. `None`).
    fn alloc_in_arena(
        tcx: TyCtxt<'tcx>,
        typed_arena: &'tcx TypedArena<Self::Allocated>,
        value: Self::Provided,
    ) -> Self;
}

impl<'tcx, T> ArenaCached<'tcx> for &'tcx T {
    type Provided = T;
    type Allocated = T;

    fn alloc_in_arena(tcx: TyCtxt<'tcx>, typed_arena: &'tcx TypedArena<T>, value: T) -> Self {
        // Just allocate in the arena normally.
        do_alloc(tcx, typed_arena, value)
    }
}

impl<'tcx, T> ArenaCached<'tcx> for Option<&'tcx T> {
    type Provided = Option<T>;
    /// The provide value is `Option<T>`, but we only store `T` in the arena.
    type Allocated = T;

    fn alloc_in_arena(
        tcx: TyCtxt<'tcx>,
        typed_arena: &'tcx TypedArena<T>,
        value: Option<T>,
    ) -> Self {
        // Don't store None in the arena, and wrap the allocated reference in Some.
        try { do_alloc(tcx, typed_arena, value?) }
    }
}

/// Allocates a value in either its dedicated arena, or in the common dropless
/// arena, depending on whether it needs to be dropped.
fn do_alloc<'tcx, T>(tcx: TyCtxt<'tcx>, typed_arena: &'tcx TypedArena<T>, value: T) -> &'tcx T {
    if mem::needs_drop::<T>() { typed_arena.alloc(value) } else { tcx.arena.dropless.alloc(value) }
}
