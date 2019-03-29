use arena::{TypedArena, DroplessArena};

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

macro_rules! impl_specialized_decodable {
    ([decode] $ty:ty, $tcx:lifetime) => {
        impl<$tcx> serialize::UseSpecializedDecodable for &$tcx $ty {}
    };
    ([] $ty:ty, $tcx:lifetime) => {};
}

macro_rules! impl_arena_allocatable {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl_specialized_decodable!($a $ty, $tcx);

            impl<$tcx> ArenaAllocatable<$tcx> for $ty {
                #[inline]
                fn arena<'a>(arena: &'a Arena<$tcx>) -> Option<&'a TypedArena<Self>> {
                    Some(&arena.$name)
                }
            }
        )*
    }
}

arena_types!(declare_arena, [], 'tcx);

arena_types!(impl_arena_allocatable, [], 'tcx);

pub trait ArenaAllocatable<'tcx>: Sized {
    /// Returns a specific arena to allocate from if the type requires destructors.
    /// Otherwise it will return `None` to be allocated from the dropless arena.
    fn arena<'a>(arena: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>>;
}

impl<'tcx, T: Copy> ArenaAllocatable<'tcx> for T {
    #[inline]
    default fn arena<'a>(_: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>> {
        None
    }
}

impl<'tcx> Arena<'tcx> {
    #[inline]
    pub fn alloc<T: ArenaAllocatable<'tcx>>(&self, value: T) -> &mut T {
        match T::arena(self) {
            Some(arena) => {
                arena.alloc(value)
            }
            None => {
                self.dropless.alloc(value)
            }
        }
    }

    pub fn alloc_from_iter<
        T: ArenaAllocatable<'tcx>,
        I: IntoIterator<Item = T>
    >(
        &self,
        iter: I
    ) -> &mut [T] {
        match T::arena(self) {
            Some(arena) => {
                arena.alloc_from_iter(iter)
            }
            None => {
                self.dropless.alloc_from_iter(iter)
            }
        }
    }
}
