use crate::alloc::Global;
use crate::fmt;
use core::alloc::{AllocError, Allocator, Layout};
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

#[cfg(test)]
mod tests;

/// Allocator that adds appropriate padding for a repr(C) struct.
///
/// This allocator takes as type arguments the type of a field `T` and an allocator `A`.
///
/// Consider
///
/// ```rust,ignore (not real code)
/// #[repr(C)]
/// struct Struct {
///     t: T,
///     data: Data,
/// }
/// ```
///
/// where `Data` is a type with layout `layout`.
///
/// When this allocator creates an allocation for layout `layout`, the pointer can be
/// offset by `-offsetof(Struct, data)` and the resulting pointer points is an allocation
/// of `A` for `Layout::new::<Struct>()`.
pub(crate) struct StructAlloc<T, A = Global>(A, PhantomData<*const T>);

impl<T, A> StructAlloc<T, A> {
    #[allow(dead_code)]
    /// Creates a new allocator.
    pub(crate) fn new(allocator: A) -> Self {
        Self(allocator, PhantomData)
    }

    /// Computes the layout of `Struct`.
    fn struct_layout(data_layout: Layout) -> Result<Layout, AllocError> {
        let t_align = mem::align_of::<T>();
        let t_size = mem::size_of::<T>();
        if t_size == 0 && t_align == 1 {
            // Skip the checks below
            return Ok(data_layout);
        }
        let data_align = data_layout.align();
        // The contract of `Layout` guarantees that `data_align > 0`.
        let data_align_minus_1 = data_align.wrapping_sub(1);
        let data_size = data_layout.size();
        let align = data_align.max(t_align);
        let align_minus_1 = align.wrapping_sub(1);
        // `size` is
        //     t_size rounded up to `data_align`
        // plus
        //     `data_size` rounded up to `align`
        // Note that the result is a multiple of `align`.
        let (t_size_aligned, t_overflow) =
            t_size.overflowing_add(t_size.wrapping_neg() & data_align_minus_1);
        let (data_size_aligned, data_overflow) = match data_size.overflowing_add(align_minus_1) {
            (sum, req_overflow) => (sum & !align_minus_1, req_overflow),
        };
        let (size, sum_overflow) = t_size_aligned.overflowing_add(data_size_aligned);
        if t_overflow || data_overflow || sum_overflow {
            return Err(AllocError);
        }
        unsafe { Ok(Layout::from_size_align_unchecked(size, align)) }
    }

    /// Returns the offset of `data` in `Struct`.
    #[inline]
    pub(crate) fn offset_of_data(data_layout: Layout) -> usize {
        let t_size = mem::size_of::<T>();
        // The contract of `Layout` guarantees `.align() > 0`
        let data_align_minus_1 = data_layout.align().wrapping_sub(1);
        t_size.wrapping_add(t_size.wrapping_neg() & data_align_minus_1)
    }

    /// Given a pointer to `data`, returns a pointer to `Struct`.
    ///
    /// # Safety
    ///
    /// The data pointer must have been allocated by `Self` with the same `data_layout`.
    #[inline]
    unsafe fn data_ptr_to_struct_ptr(data: NonNull<u8>, data_layout: Layout) -> NonNull<u8> {
        unsafe {
            let offset_of_data = Self::offset_of_data(data_layout);
            NonNull::new_unchecked(data.as_ptr().sub(offset_of_data))
        }
    }

    /// Given a pointer to `Struct`, returns a pointer to `data`.
    ///
    /// # Safety
    ///
    /// The struct pointer must have been allocated by `A` with the layout
    /// `Self::struct_layout(data_layout)`.
    #[inline]
    unsafe fn struct_ptr_to_data_ptr(
        struct_ptr: NonNull<[u8]>,
        data_layout: Layout,
    ) -> NonNull<[u8]> {
        let offset_of_data = Self::offset_of_data(data_layout);
        let data_ptr =
            unsafe { NonNull::new_unchecked(struct_ptr.as_mut_ptr().add(offset_of_data)) };
        // Note that the size is the exact size requested in the layout. Let me explain
        // why this is necessary:
        //
        // Assume the original requested layout was `size=1, align=1`. Assume `T=u16`
        // Then the struct layout is `size=4, align=2`. Assume that the allocator returns
        // a slice with `size=5`. Then the space available for `data` is `3`.
        // However, if we returned a slice with `len=3`, then the user would be allowed
        // to call `dealloc` with `size=3, align=1`. In this case the struct layout
        // would be computed as `size=6, align=2`. This size would be larger than what
        // the allocator returned.
        NonNull::slice_from_raw_parts(data_ptr, data_layout.size())
    }
}

impl<T, A: Debug> Debug for StructAlloc<T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructAlloc").field("0", &self.0).finish()
    }
}

/// Delegates `Self::allocate{,_zereod}` to the allocator after computing the struct
/// layout. Then transforms the new struct pointer to the new data pointer and returns it.
macro delegate_alloc($id:ident) {
    fn $id(&self, data_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let struct_layout = Self::struct_layout(data_layout)?;
        let struct_ptr = self.0.$id(struct_layout)?;
        unsafe { Ok(Self::struct_ptr_to_data_ptr(struct_ptr, data_layout)) }
    }
}

/// Delegates `Self::{{grow{,_zeroed},shrink}` to the allocator after computing the struct
/// layout and transforming the data pointer to the struct pointer. Then transforms
/// the new struct pointer to the new data pointer and returns it.
macro delegate_transform($id:ident) {
    unsafe fn $id(
        &self,
        old_data_ptr: NonNull<u8>,
        old_data_layout: Layout,
        new_data_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_struct_layout = Self::struct_layout(old_data_layout)?;
        let new_struct_layout = Self::struct_layout(new_data_layout)?;
        unsafe {
            let old_struct_ptr = Self::data_ptr_to_struct_ptr(old_data_ptr, old_data_layout);
            let new_struct_ptr =
                self.0.$id(old_struct_ptr, old_struct_layout, new_struct_layout)?;
            Ok(Self::struct_ptr_to_data_ptr(new_struct_ptr, new_data_layout))
        }
    }
}

unsafe impl<T, A: Allocator> Allocator for StructAlloc<T, A> {
    delegate_alloc!(allocate);
    delegate_alloc!(allocate_zeroed);

    unsafe fn deallocate(&self, data_ptr: NonNull<u8>, data_layout: Layout) {
        unsafe {
            let struct_ptr = Self::data_ptr_to_struct_ptr(data_ptr, data_layout);
            let struct_layout =
                Self::struct_layout(data_layout).expect("deallocate called with invalid layout");
            self.0.deallocate(struct_ptr, struct_layout);
        }
    }

    delegate_transform!(grow);
    delegate_transform!(grow_zeroed);
    delegate_transform!(shrink);
}

#[allow(unused_macros)]
macro_rules! implement_struct_allocator {
    ($id:ident) => {
        #[unstable(feature = "struct_alloc", issue = "none")]
        unsafe impl<A: Allocator> Allocator for $id<A> {
            fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                self.0.allocate(layout)
            }

            fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                self.0.allocate_zeroed(layout)
            }

            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                unsafe { self.0.deallocate(ptr, layout) }
            }

            unsafe fn grow(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                unsafe { self.0.grow(ptr, old_layout, new_layout) }
            }

            unsafe fn grow_zeroed(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                unsafe { self.0.grow_zeroed(ptr, old_layout, new_layout) }
            }

            unsafe fn shrink(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                unsafe { self.0.shrink(ptr, old_layout, new_layout) }
            }
        }
    };
}
