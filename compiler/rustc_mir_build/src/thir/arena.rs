use crate::thir::*;

macro_rules! declare_arena {
    ([], [$($a:tt $name:ident: $ty:ty,)*]) => {
        #[derive(Default)]
        pub struct Arena<'thir, 'tcx> {
            pub dropless: rustc_arena::DroplessArena,
            drop: rustc_arena::DropArena,
            $($name: rustc_arena::arena_for_type!($a[$ty]),)*
        }

        pub trait ArenaAllocatable<'thir, 'tcx, T = Self>: Sized {
            fn allocate_on(self, arena: &'thir Arena<'thir, 'tcx>) -> &'thir mut Self;
            fn allocate_from_iter(
                arena: &'thir Arena<'thir, 'tcx>,
                iter: impl ::std::iter::IntoIterator<Item = Self>,
            ) -> &'thir mut [Self];
        }

        impl<'thir, 'tcx, T: Copy> ArenaAllocatable<'thir, 'tcx, ()> for T {
            #[inline]
            fn allocate_on(self, arena: &'thir Arena<'thir, 'tcx>) -> &'thir mut Self {
                arena.dropless.alloc(self)
            }
            #[inline]
            fn allocate_from_iter(
                arena: &'thir Arena<'thir, 'tcx>,
                iter: impl ::std::iter::IntoIterator<Item = Self>,
            ) -> &'thir mut [Self] {
                arena.dropless.alloc_from_iter(iter)
            }

        }
        $(
            impl<'thir, 'tcx> ArenaAllocatable<'thir, 'tcx, $ty> for $ty {
                #[inline]
                fn allocate_on(self, arena: &'thir Arena<'thir, 'tcx>) -> &'thir mut Self {
                    if !::std::mem::needs_drop::<Self>() {
                        return arena.dropless.alloc(self);
                    }
                    match rustc_arena::which_arena_for_type!($a[&arena.$name]) {
                        ::std::option::Option::<&rustc_arena::TypedArena<Self>>::Some(ty_arena) => {
                            ty_arena.alloc(self)
                        }
                        ::std::option::Option::None => unsafe { arena.drop.alloc(self) },
                    }
                }

                #[inline]
                fn allocate_from_iter(
                    arena: &'thir Arena<'thir, 'tcx>,
                    iter: impl ::std::iter::IntoIterator<Item = Self>,
                ) -> &'thir mut [Self] {
                    if !::std::mem::needs_drop::<Self>() {
                        return arena.dropless.alloc_from_iter(iter);
                    }
                    match rustc_arena::which_arena_for_type!($a[&arena.$name]) {
                        ::std::option::Option::<&rustc_arena::TypedArena<Self>>::Some(ty_arena) => {
                            ty_arena.alloc_from_iter(iter)
                        }
                        ::std::option::Option::None => unsafe { arena.drop.alloc_from_iter(iter) },
                    }
                }
            }
        )*

        impl<'thir, 'tcx> Arena<'thir, 'tcx> {
            #[inline]
            pub fn alloc<T: ArenaAllocatable<'thir, 'tcx, U>, U>(&'thir self, value: T) -> &'thir mut T {
                value.allocate_on(self)
            }

            #[allow(dead_code)] // function is never used
            #[inline]
            pub fn alloc_slice<T: ::std::marker::Copy>(&'thir self, value: &[T]) -> &'thir mut [T] {
                if value.is_empty() {
                    return &mut [];
                }
                self.dropless.alloc_slice(value)
            }

            pub fn alloc_from_iter<T: ArenaAllocatable<'thir, 'tcx, U>, U>(
                &'thir self,
                iter: impl ::std::iter::IntoIterator<Item = T>,
            ) -> &'thir mut [T] {
                T::allocate_from_iter(self, iter)
            }
        }
    }
}

declare_arena!([], [
    [] arm: Arm<'thir, 'tcx>,
    [] expr: Expr<'thir, 'tcx>,
    [] field_expr: FieldExpr<'thir, 'tcx>,
    [few] inline_asm_operand: InlineAsmOperand<'thir, 'tcx>,
    [] stmt: Stmt<'thir, 'tcx>,
]);
