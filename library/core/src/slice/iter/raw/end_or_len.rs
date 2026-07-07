use crate::mem::SizedTypeProperties;

pub(super) struct EndOrLenRepr<T>(*const T);

/// A view into [`IterRaw::end_or_len`] field.
///
/// If rust allowed using `T::IS_ZST` in the type system, we could make this the actual type of
/// the field (by making `End` have an uninhabited type if `T::IS_ZST` and vice-versa for `Len`).
///
/// But, as it stands, we have to resort to using this only as a more convinient way of accessing
/// the field, and depend on inlining to figure out that for a given `T`, only one variant is
/// possible...
pub(super) enum EndOrLenView<E, L> {
    End(E),
    Len(L),
}

pub(super) use EndOrLenView::*;

use crate::ptr::{self, NonNull};

pub(super) type EndOrLen<T> = EndOrLenView<NonNull<T>, usize>;
pub(super) type EndOrLenMut<'a, T> = EndOrLenView<&'a mut NonNull<T>, &'a mut usize>;

impl<T> EndOrLenRepr<T> {
    #[rustc_force_inline]
    #[inline(always)]
    pub(super) const fn new(zst: usize, non_zst: NonNull<T>) -> Self {
        if T::IS_ZST {
            EndOrLenRepr(ptr::without_provenance(zst))
        } else {
            EndOrLenRepr(non_zst.as_ptr())
        }
    }

    #[rustc_force_inline]
    #[inline(always)]
    pub(super) fn view(&self) -> EndOrLen<T> {
        if T::IS_ZST {
            Len(self.0.addr())
        } else {
            unsafe { End(NonNull::new_unchecked(self.0.cast_mut())) }
        }
    }

    #[rustc_force_inline]
    #[inline(always)]
    pub(super) fn view_mut(&mut self) -> EndOrLenMut<'_, T> {
        if T::IS_ZST {
            Len(unsafe { &mut *ptr::from_mut(&mut self.0).cast::<usize>() })
        } else {
            End(unsafe { &mut *ptr::from_mut(&mut self.0).cast::<NonNull<T>>() })
        }
    }
}

impl<T> Copy for EndOrLenRepr<T> {}
impl<T> Clone for EndOrLenRepr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
