use arena::{TypedArena, DroplessArena};
use std::mem;

#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            [] vtable_method: Option<(
                rustc::hir::def_id::DefId,
                rustc::ty::subst::SubstsRef<$tcx>
            )>,
            [decode] specialization_graph: rustc::traits::specialization_graph::Graph,
        ], $tcx);
    )
}

macro_rules! declare_arena {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        #[derive(Default)]
        pub struct Arena<$tcx> {
            dropless: DroplessArena,
            $($name: TypedArena<$ty>,)*
        }
    }
}

macro_rules! impl_arena_allocatable {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl ArenaAllocatable for $ty {}
            impl<$tcx> ArenaField<$tcx> for $ty {
                #[inline]
                fn arena<'a>(arena: &'a Arena<$tcx>) -> &'a TypedArena<Self> {
                    &arena.$name
                }
            }
        )*
    }
}

arena_types!(declare_arena, [], 'tcx);

arena_types!(impl_arena_allocatable, [], 'tcx);

pub trait ArenaAllocatable {}

impl<T: Copy> ArenaAllocatable for T {}

pub trait ArenaField<'tcx>: Sized {
    /// Returns a specific arena to allocate from.
    fn arena<'a>(arena: &'a Arena<'tcx>) -> &'a TypedArena<Self>;
}

impl<'tcx, T> ArenaField<'tcx> for T {
    #[inline]
    default fn arena<'a>(_: &'a Arena<'tcx>) -> &'a TypedArena<Self> {
        panic!()
    }
}

impl<'tcx> Arena<'tcx> {
    #[inline]
    pub fn alloc<T: ArenaAllocatable>(&self, value: T) -> &mut T {
        if mem::needs_drop::<T>() {
            <T as ArenaField<'tcx>>::arena(self).alloc(value)
        } else {
            self.dropless.alloc(value)
        }
    }

    pub fn alloc_from_iter<
        T: ArenaAllocatable,
        I: IntoIterator<Item = T>
    >(
        &self,
        iter: I
    ) -> &mut [T] {
        if mem::needs_drop::<T>() {
            <T as ArenaField<'tcx>>::arena(self).alloc_from_iter(iter)
        } else {
            self.dropless.alloc_from_iter(iter)
        }
    }
}
