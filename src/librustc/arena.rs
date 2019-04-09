use arena::{TypedArena, DroplessArena};
use std::mem;
use std::ptr;
use std::slice;
use std::cell::RefCell;
use std::marker::PhantomData;
use smallvec::SmallVec;

#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            [] vtable_method: Option<(
                rustc::hir::def_id::DefId,
                rustc::ty::subst::SubstsRef<$tcx>
            )>,
            [few] mir_keys: rustc::util::nodemap::DefIdSet,
            [decode] specialization_graph: rustc::traits::specialization_graph::Graph,
        ], $tcx);
    )
}

macro_rules! arena_for_type {
    ([][$ty:ty]) => {
        TypedArena<$ty>
    };
    ([few $(, $attrs:ident)*][$ty:ty]) => {
        PhantomData<$ty>
    };
    ([$ignore:ident $(, $attrs:ident)*]$args:tt) => {
        arena_for_type!([$($attrs),*]$args)
    };
}

macro_rules! declare_arena {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        #[derive(Default)]
        pub struct Arena<$tcx> {
            dropless: DroplessArena,
            drop: DropArena,
            $($name: arena_for_type!($a[$ty]),)*
        }
    }
}

macro_rules! which_arena_for_type {
    ([][$arena:expr]) => {
        Some($arena)
    };
    ([few$(, $attrs:ident)*][$arena:expr]) => {
        None
    };
    ([$ignore:ident$(, $attrs:ident)*]$args:tt) => {
        which_arena_for_type!([$($attrs),*]$args)
    };
}

macro_rules! impl_arena_allocatable {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        $(
            impl ArenaAllocatable for $ty {}
            unsafe impl<$tcx> ArenaField<$tcx> for $ty {
                #[inline]
                fn arena<'a>(_arena: &'a Arena<$tcx>) -> Option<&'a TypedArena<Self>> {
                    which_arena_for_type!($a[&_arena.$name])
                }
            }
        )*
    }
}

arena_types!(declare_arena, [], 'tcx);

arena_types!(impl_arena_allocatable, [], 'tcx);

pub trait ArenaAllocatable {}

impl<T: Copy> ArenaAllocatable for T {}

pub unsafe trait ArenaField<'tcx>: Sized {
    /// Returns a specific arena to allocate from.
    /// If None is returned, the DropArena will be used.
    fn arena<'a>(arena: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>>;
}

unsafe impl<'tcx, T> ArenaField<'tcx> for T {
    #[inline]
    default fn arena<'a>(_: &'a Arena<'tcx>) -> Option<&'a TypedArena<Self>> {
        panic!()
    }
}

impl<'tcx> Arena<'tcx> {
    #[inline]
    pub fn alloc<T: ArenaAllocatable>(&self, value: T) -> &mut T {
        if !mem::needs_drop::<T>() {
            return self.dropless.alloc(value);
        }
        match <T as ArenaField<'tcx>>::arena(self) {
            Some(arena) => arena.alloc(value),
            None => unsafe { self.drop.alloc(value) },
        }
    }

    pub fn alloc_from_iter<
        T: ArenaAllocatable,
        I: IntoIterator<Item = T>
    >(
        &'a self,
        iter: I
    ) -> &'a mut [T] {
        if !mem::needs_drop::<T>() {
            return self.dropless.alloc_from_iter(iter);
        }
        match <T as ArenaField<'tcx>>::arena(self) {
            Some(arena) => arena.alloc_from_iter(iter),
            None => unsafe { self.drop.alloc_from_iter(iter) },
        }
    }
}

/// Calls the destructor for an object when dropped.
struct DropType {
    drop_fn: unsafe fn(*mut u8),
    obj: *mut u8,
}

unsafe fn drop_for_type<T>(to_drop: *mut u8) {
    std::ptr::drop_in_place(to_drop as *mut T)
}

impl Drop for DropType {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.obj)
        }
    }
}

/// An arena which can be used to allocate any type.
/// Allocating in this arena is unsafe since the type system
/// doesn't know which types it contains. In order to
/// allocate safely, you must store a PhantomData<T>
/// alongside this arena for each type T you allocate.
#[derive(Default)]
struct DropArena {
    /// A list of destructors to run when the arena drops.
    /// Ordered so `destructors` gets dropped before the arena
    /// since its destructor can reference memory in the arena.
    destructors: RefCell<Vec<DropType>>,
    arena: DroplessArena,
}

impl DropArena {
    #[inline]
    unsafe fn alloc<T>(&self, object: T) -> &mut T {
        let mem = self.arena.alloc_raw(
            mem::size_of::<T>(),
            mem::align_of::<T>()
        ) as *mut _ as *mut T;
        // Write into uninitialized memory.
        ptr::write(mem, object);
        let result = &mut *mem;
        // Record the destructor after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        self.destructors.borrow_mut().push(DropType {
            drop_fn: drop_for_type::<T>,
            obj: result as *mut T as *mut u8,
        });
        result
    }

    #[inline]
    unsafe fn alloc_from_iter<T, I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        let mut vec: SmallVec<[_; 8]> = iter.into_iter().collect();
        if vec.is_empty() {
            return &mut [];
        }
        let len = vec.len();

        let start_ptr = self.arena.alloc_raw(
            len.checked_mul(mem::size_of::<T>()).unwrap(),
            mem::align_of::<T>()
        ) as *mut _ as *mut T;

        let mut destructors = self.destructors.borrow_mut();
        // Reserve space for the destructors so we can't panic while adding them
        destructors.reserve(len);

        // Move the content to the arena by copying it and then forgetting
        // the content of the SmallVec
        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
        mem::forget(vec.drain());

        // Record the destructors after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        for i in 0..len {
            destructors.push(DropType {
                drop_fn: drop_for_type::<T>,
                obj: start_ptr.offset(i as isize) as *mut u8,
            });
        }

        slice::from_raw_parts_mut(start_ptr, len)
    }
}
