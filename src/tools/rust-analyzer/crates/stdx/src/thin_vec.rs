use std::alloc::{dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::{addr_of_mut, slice_from_raw_parts_mut, NonNull};

/// A type that is functionally equivalent to `(Header, Box<[Item]>)`,
/// but all data is stored in one heap allocation and the pointer is thin,
/// so the whole thing's size is like a pointer.
pub struct ThinVecWithHeader<Header, Item> {
    /// INVARIANT: Points to a valid heap allocation that contains `ThinVecInner<Header>`,
    /// followed by (suitably aligned) `len` `Item`s.
    ptr: NonNull<ThinVecInner<Header>>,
    _marker: PhantomData<(Header, Box<[Item]>)>,
}

// SAFETY: We essentially own both the header and the items.
unsafe impl<Header: Send, Item: Send> Send for ThinVecWithHeader<Header, Item> {}
unsafe impl<Header: Sync, Item: Sync> Sync for ThinVecWithHeader<Header, Item> {}

#[derive(Clone)]
struct ThinVecInner<Header> {
    header: Header,
    len: usize,
}

impl<Header, Item> ThinVecWithHeader<Header, Item> {
    /// # Safety
    ///
    /// The iterator must produce `len` elements.
    #[inline]
    unsafe fn from_trusted_len_iter(
        header: Header,
        len: usize,
        items: impl Iterator<Item = Item>,
    ) -> Self {
        let (ptr, layout, items_offset) = Self::allocate(len);

        struct DeallocGuard(*mut u8, Layout);
        impl Drop for DeallocGuard {
            fn drop(&mut self) {
                // SAFETY: We allocated this above.
                unsafe {
                    dealloc(self.0, self.1);
                }
            }
        }
        let _dealloc_guard = DeallocGuard(ptr.as_ptr().cast::<u8>(), layout);

        // INVARIANT: Between `0..1` there are only initialized items.
        struct ItemsGuard<Item>(*mut Item, *mut Item);
        impl<Item> Drop for ItemsGuard<Item> {
            fn drop(&mut self) {
                // SAFETY: Our invariant.
                unsafe {
                    slice_from_raw_parts_mut(self.0, self.1.offset_from(self.0) as usize)
                        .drop_in_place();
                }
            }
        }

        // SAFETY: We allocated enough space.
        let mut items_ptr = unsafe { ptr.as_ptr().byte_add(items_offset).cast::<Item>() };
        // INVARIANT: There are zero elements in this range.
        let mut items_guard = ItemsGuard(items_ptr, items_ptr);
        items.for_each(|item| {
            // SAFETY: Our precondition guarantee we won't get more than `len` items, and we allocated
            // enough space for `len` items.
            unsafe {
                items_ptr.write(item);
                items_ptr = items_ptr.add(1);
            }
            // INVARIANT: We just initialized this item.
            items_guard.1 = items_ptr;
        });

        // SAFETY: We allocated enough space.
        unsafe {
            ptr.write(ThinVecInner { header, len });
        }

        std::mem::forget(items_guard);

        std::mem::forget(_dealloc_guard);

        // INVARIANT: We allocated and initialized all fields correctly.
        Self { ptr, _marker: PhantomData }
    }

    #[inline]
    fn allocate(len: usize) -> (NonNull<ThinVecInner<Header>>, Layout, usize) {
        let (layout, items_offset) = Self::layout(len);
        // SAFETY: We always have `len`, so our allocation cannot be zero-sized.
        let ptr = unsafe { std::alloc::alloc(layout).cast::<ThinVecInner<Header>>() };
        let Some(ptr) = NonNull::<ThinVecInner<Header>>::new(ptr) else {
            handle_alloc_error(layout);
        };
        (ptr, layout, items_offset)
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(header: Header, items: I) -> Self
    where
        I: IntoIterator,
        I::IntoIter: TrustedLen<Item = Item>,
    {
        let items = items.into_iter();
        // SAFETY: `TrustedLen` guarantees the iterator length is exact.
        unsafe { Self::from_trusted_len_iter(header, items.len(), items) }
    }

    #[inline]
    fn items_offset(&self) -> usize {
        // SAFETY: We `pad_to_align()` in `layout()`, so at most where accessing past the end of the allocation,
        // which is allowed.
        unsafe {
            Layout::new::<ThinVecInner<Header>>().extend(Layout::new::<Item>()).unwrap_unchecked().1
        }
    }

    #[inline]
    fn header_and_len(&self) -> &ThinVecInner<Header> {
        // SAFETY: By `ptr`'s invariant, it is correctly allocated and initialized.
        unsafe { &*self.ptr.as_ptr() }
    }

    #[inline]
    fn items_ptr(&self) -> *mut [Item] {
        let len = self.header_and_len().len;
        // SAFETY: `items_offset()` returns the correct offset of the items, where they are allocated.
        let ptr = unsafe { self.ptr.as_ptr().byte_add(self.items_offset()).cast::<Item>() };
        slice_from_raw_parts_mut(ptr, len)
    }

    #[inline]
    pub fn header(&self) -> &Header {
        &self.header_and_len().header
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut Header {
        // SAFETY: By `ptr`'s invariant, it is correctly allocated and initialized.
        unsafe { &mut *addr_of_mut!((*self.ptr.as_ptr()).header) }
    }

    #[inline]
    pub fn items(&self) -> &[Item] {
        // SAFETY: `items_ptr()` gives a valid pointer.
        unsafe { &*self.items_ptr() }
    }

    #[inline]
    pub fn items_mut(&mut self) -> &mut [Item] {
        // SAFETY: `items_ptr()` gives a valid pointer.
        unsafe { &mut *self.items_ptr() }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.header_and_len().len
    }

    #[inline]
    fn layout(len: usize) -> (Layout, usize) {
        let (layout, items_offset) = Layout::new::<ThinVecInner<Header>>()
            .extend(Layout::array::<Item>(len).expect("too big `ThinVec` requested"))
            .expect("too big `ThinVec` requested");
        let layout = layout.pad_to_align();
        (layout, items_offset)
    }
}

/// # Safety
///
/// The length reported must be exactly the number of items yielded.
pub unsafe trait TrustedLen: ExactSizeIterator {}

unsafe impl<T> TrustedLen for std::vec::IntoIter<T> {}
unsafe impl<T> TrustedLen for std::slice::Iter<'_, T> {}
unsafe impl<'a, T: Clone + 'a, I: TrustedLen<Item = &'a T>> TrustedLen for std::iter::Cloned<I> {}
unsafe impl<T, I: TrustedLen, F: FnMut(I::Item) -> T> TrustedLen for std::iter::Map<I, F> {}
unsafe impl<T> TrustedLen for std::vec::Drain<'_, T> {}
unsafe impl<T, const N: usize> TrustedLen for std::array::IntoIter<T, N> {}

impl<Header: Clone, Item: Clone> Clone for ThinVecWithHeader<Header, Item> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_iter(self.header().clone(), self.items().iter().cloned())
    }
}

impl<Header, Item> Drop for ThinVecWithHeader<Header, Item> {
    #[inline]
    fn drop(&mut self) {
        // This must come before we drop `header`, because after that we cannot make a reference to it in `len()`.
        let len = self.len();

        // SAFETY: The contents are allocated and initialized.
        unsafe {
            addr_of_mut!((*self.ptr.as_ptr()).header).drop_in_place();
            self.items_ptr().drop_in_place();
        }

        let (layout, _) = Self::layout(len);
        // SAFETY: This was allocated in `new()` with the same layout calculation.
        unsafe {
            dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
        }
    }
}

impl<Header: fmt::Debug, Item: fmt::Debug> fmt::Debug for ThinVecWithHeader<Header, Item> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThinVecWithHeader")
            .field("header", self.header())
            .field("items", &self.items())
            .finish()
    }
}

impl<Header: PartialEq, Item: PartialEq> PartialEq for ThinVecWithHeader<Header, Item> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.items() == other.items()
    }
}

impl<Header: Eq, Item: Eq> Eq for ThinVecWithHeader<Header, Item> {}

impl<Header: Hash, Item: Hash> Hash for ThinVecWithHeader<Header, Item> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.header().hash(state);
        self.items().hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ThinVec<T>(ThinVecWithHeader<(), T>);

impl<T> ThinVec<T> {
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(values: I) -> Self
    where
        I: IntoIterator,
        I::IntoIter: TrustedLen<Item = T>,
    {
        Self(ThinVecWithHeader::from_iter((), values))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        (**self).iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        (**self).iter_mut()
    }
}

impl<T> Deref for ThinVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.items()
    }
}

impl<T> DerefMut for ThinVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.items_mut()
    }
}

impl<'a, T> IntoIterator for &'a ThinVec<T> {
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ThinVec<T> {
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for ThinVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&**self).finish()
    }
}

/// A [`ThinVec`] that requires no allocation for the empty case.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EmptyOptimizedThinVec<T>(Option<ThinVec<T>>);

impl<T> EmptyOptimizedThinVec<T> {
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(values: I) -> Self
    where
        I: IntoIterator,
        I::IntoIter: TrustedLen<Item = T>,
    {
        let values = values.into_iter();
        if values.len() == 0 {
            Self::empty()
        } else {
            Self(Some(ThinVec::from_iter(values)))
        }
    }

    #[inline]
    pub fn empty() -> Self {
        Self(None)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, ThinVec::len)
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        (**self).iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        (**self).iter_mut()
    }
}

impl<T> Default for EmptyOptimizedThinVec<T> {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> Deref for EmptyOptimizedThinVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or_default()
    }
}

impl<T> DerefMut for EmptyOptimizedThinVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_deref_mut().unwrap_or_default()
    }
}

impl<'a, T> IntoIterator for &'a EmptyOptimizedThinVec<T> {
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut EmptyOptimizedThinVec<T> {
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for EmptyOptimizedThinVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&**self).finish()
    }
}

/// Syntax:
///
/// ```ignore
/// thin_vec_with_header_struct! {
///     pub new(pub(crate)) struct MyCoolStruct, MyCoolStructHeader {
///         pub(crate) variable_length: [Ty],
///         pub field1: CopyTy,
///         pub field2: NonCopyTy; ref,
///     }
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! thin_vec_with_header_struct_ {
    (@maybe_ref (ref) $($t:tt)*) => { &$($t)* };
    (@maybe_ref () $($t:tt)*) => { $($t)* };
    (
        $vis:vis new($new_vis:vis) struct $struct:ident, $header:ident {
            $items_vis:vis $items:ident : [$items_ty:ty],
            $( $header_var_vis:vis $header_var:ident : $header_var_ty:ty $(; $ref:ident)?, )+
        }
    ) => {
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        struct $header {
            $( $header_var : $header_var_ty, )+
        }

        #[derive(Clone, Eq, PartialEq, Hash)]
        $vis struct $struct($crate::thin_vec::ThinVecWithHeader<$header, $items_ty>);

        impl $struct {
            #[inline]
            #[allow(unused)]
            $new_vis fn new<I>(
                $( $header_var: $header_var_ty, )+
                $items: I,
            ) -> Self
            where
                I: ::std::iter::IntoIterator,
                I::IntoIter: $crate::thin_vec::TrustedLen<Item = $items_ty>,
            {
                Self($crate::thin_vec::ThinVecWithHeader::from_iter(
                    $header { $( $header_var, )+ },
                    $items,
                ))
            }

            #[inline]
            $items_vis fn $items(&self) -> &[$items_ty] {
                self.0.items()
            }

            $(
                #[inline]
                $header_var_vis fn $header_var(&self) -> $crate::thin_vec_with_header_struct_!(@maybe_ref ($($ref)?) $header_var_ty) {
                    $crate::thin_vec_with_header_struct_!(@maybe_ref ($($ref)?) self.0.header().$header_var)
                }
            )+
        }

        impl ::std::fmt::Debug for $struct {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_struct(stringify!($struct))
                    $( .field(stringify!($header_var), &self.$header_var()) )*
                    .field(stringify!($items), &self.$items())
                    .finish()
            }
        }
    };
}
pub use crate::thin_vec_with_header_struct_ as thin_vec_with_header_struct;
