#![allow(unused)]

use fortanix_sgx_abi::*;

use super::super::mem::{is_enclave_range, is_user_range};
use crate::arch::asm;
use crate::cell::UnsafeCell;
use crate::convert::TryInto;
use crate::marker::{PhantomData, Unsize};
use crate::mem::{self, ManuallyDrop, MaybeUninit};
use crate::ops::{CoerceUnsized, Deref, DerefMut, Index, IndexMut};
use crate::ptr::{self, NonNull};
use crate::slice::SliceIndex;
use crate::{cmp, intrinsics, slice};

/// A type that can be safely read from or written to userspace.
///
/// Non-exhaustive list of specific requirements for reading and writing:
/// * **Type is `Copy`** (and therefore also not `Drop`). Copies will be
///   created when copying from/to userspace. Destructors will not be called.
/// * **No references or Rust-style owned pointers** (`Vec`, `Arc`, etc.). When
///   reading from userspace, references into enclave memory must not be
///   created. Also, only enclave memory is considered managed by the Rust
///   compiler's static analysis. When reading from userspace, there can be no
///   guarantee that the value correctly adheres to the expectations of the
///   type. When writing to userspace, memory addresses of data in enclave
///   memory must not be leaked for confidentiality reasons. `User` and
///   `UserRef` are also not allowed for the same reasons.
/// * **No fat pointers.** When reading from userspace, the size or vtable
///   pointer could be automatically interpreted and used by the code. When
///   writing to userspace, memory addresses of data in enclave memory (such
///   as vtable pointers) must not be leaked for confidentiality reasons.
///
/// Non-exhaustive list of specific requirements for reading from userspace:
/// * **Any bit pattern is valid** for this type (no `enum`s). There can be no
///   guarantee that the value correctly adheres to the expectations of the
///   type, so any value must be valid for this type.
///
/// Non-exhaustive list of specific requirements for writing to userspace:
/// * **No pointers to enclave memory.** Memory addresses of data in enclave
///   memory must not be leaked for confidentiality reasons.
/// * **No internal padding.** Padding might contain previously-initialized
///   secret data stored at that memory location and must not be leaked for
///   confidentiality reasons.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub unsafe trait UserSafeSized: Copy + Sized {}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl UserSafeSized for u8 {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T> UserSafeSized for FifoDescriptor<T> {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl UserSafeSized for ByteBuffer {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl UserSafeSized for Usercall {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl UserSafeSized for Return {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl UserSafeSized for Cancel {}
#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafeSized> UserSafeSized for [T; 2] {}

/// A type that can be represented in memory as one or more `UserSafeSized`s.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub unsafe trait UserSafe {
    /// Equivalent to `align_of::<Self>`.
    fn align_of() -> usize;

    /// Constructs a pointer to `Self` given a memory range in user space.
    ///
    /// N.B., this takes a size, not a length!
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory range is in user memory, is the
    /// correct size and is correctly aligned and points to the right type.
    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self;

    /// Constructs a pointer to `Self` given a memory range.
    ///
    /// N.B., this takes a size, not a length!
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory range points to the correct type.
    ///
    /// # Panics
    ///
    /// This function panics if:
    ///
    /// * the pointer is not aligned.
    /// * the pointer is null.
    /// * the pointed-to range does not fit in the address space.
    /// * the pointed-to range is not in user memory.
    unsafe fn from_raw_sized(ptr: *mut u8, size: usize) -> NonNull<Self> {
        assert!(ptr.wrapping_add(size) >= ptr);
        // SAFETY: The caller has guaranteed the pointer is valid
        let ret = unsafe { Self::from_raw_sized_unchecked(ptr, size) };
        unsafe {
            Self::check_ptr(ret);
            NonNull::new_unchecked(ret as _)
        }
    }

    /// Checks if a pointer may point to `Self` in user memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory range points to the correct type and
    /// length (if this is a slice).
    ///
    /// # Panics
    ///
    /// This function panics if:
    ///
    /// * the pointer is not aligned.
    /// * the pointer is null.
    /// * the pointed-to range is not in user memory.
    unsafe fn check_ptr(ptr: *const Self) {
        let is_aligned = |p: *const u8| -> bool { p.is_aligned_to(Self::align_of()) };

        assert!(is_aligned(ptr as *const u8));
        assert!(is_user_range(ptr as _, unsafe { mem::size_of_val_raw(ptr) }));
        assert!(!ptr.is_null());
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafeSized> UserSafe for T {
    fn align_of() -> usize {
        align_of::<T>()
    }

    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self {
        assert_eq!(size, size_of::<T>());
        ptr as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafeSized> UserSafe for [T] {
    fn align_of() -> usize {
        align_of::<T>()
    }

    /// # Panics
    ///
    /// This function panics if:
    ///
    /// * the element size is not a factor of the size
    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self {
        let elem_size = size_of::<T>();
        assert_eq!(size % elem_size, 0);
        let len = size / elem_size;
        ptr.cast::<T>().cast_slice(len)
    }
}

/// An immutable pointer to some type in userspace memory, used for reading from said memory.
///
/// `UserRef<'a, T>` is equivalent to `&'a T` in enclave memory. Access to the
/// memory is only allowed by copying to avoid TOCTTOU issues. After copying,
/// code should make sure to completely check the value before use.
///
/// It is impossible to safely obtain a direct reference to the contents of `UserRef<T>`.
/// Userspace may write to the backing memory at any time, so it can't be assumed that
/// the pointed-to memory is safely borrowed.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct UserRef<'a, T: 'a + UserSafe + ?Sized> {
    data: NonNull<T>,
    _phantom: PhantomData<&'a T>,
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe + ?Sized> Clone for UserRef<'a, T> {
    fn clone(&self) -> Self {
        unsafe { UserRef::from_non_null_unchecked(self.data) }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe + ?Sized> Copy for UserRef<'a, T> {}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<'a, T: UserSafe + ?Sized> Send for UserRef<'a, T> {}

/// Mutable equivalent to [`UserRef`]: a mutable pointer to some type in userspace memory,
/// used for writing to said memory.
///
/// `UserMut<'a, T>` is equivalent to `&'a mut T` in enclave memory. It has the same copy-out
/// and safety considerations as [`UserRef`].
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct UserMut<'a, T: 'a + UserSafe + ?Sized> {
    data: NonNull<T>,
    _phantom: PhantomData<&'a mut T>,
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<'a, T: UserSafe + ?Sized> Send for UserMut<'a, T> {}

/// An owned type in userspace memory. `User<T>` is equivalent to `Box<T>` in
/// enclave memory. Access to the memory is only allowed by copying to avoid
/// TOCTTOU issues. The user memory will be freed when the value is dropped.
/// After copying, code should make sure to completely check the value before
/// use.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct User<T: UserSafe + ?Sized>(NonNull<T>);

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafe + ?Sized> Send for User<T> {}

/// A type which can be a destination for safely copying from userspace.
///
/// # Safety
///
/// Requires that `T` and `Self` have identical layouts.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub unsafe trait UserSafeCopyDestination<T: ?Sized> {
    /// Returns a pointer for writing to the value.
    fn as_mut_ptr(&mut self) -> *mut T;
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T> UserSafeCopyDestination<T> for T {
    fn as_mut_ptr(&mut self) -> *mut T {
        self as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T> UserSafeCopyDestination<[T]> for [T] {
    fn as_mut_ptr(&mut self) -> *mut [T] {
        self as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T> UserSafeCopyDestination<T> for MaybeUninit<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut Self as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T> UserSafeCopyDestination<[T]> for [MaybeUninit<T>] {
    fn as_mut_ptr(&mut self) -> *mut [T] {
        self as *mut Self as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> User<T>
where
    T: UserSafe,
{
    // This function returns memory that is practically uninitialized, but is
    // not considered "unspecified" or "undefined" for purposes of an
    // optimizing compiler. This is achieved by returning a pointer from
    // from outside as obtained by `super::alloc`.
    fn new_uninit_bytes(size: usize) -> Self {
        unsafe {
            // Mustn't call alloc with size 0.
            let ptr = if size > 0 {
                // `copy_to_userspace` is more efficient when data is 8-byte aligned
                let alignment = cmp::max(T::align_of(), 8);
                rtunwrap!(Ok, super::alloc(size, alignment)) as _
            } else {
                T::align_of() as _ // dangling pointer ok for size 0
            };
            if let Ok(v) = crate::panic::catch_unwind(|| T::from_raw_sized(ptr, size)) {
                User(v)
            } else {
                rtabort!("Got invalid pointer from alloc() usercall")
            }
        }
    }

    /// Copies `val` into freshly allocated space in user memory.
    pub fn new_from_enclave(val: &T) -> Self {
        unsafe {
            let mut user = Self::new_uninit_bytes(size_of_val(val));
            user.as_user_mut().copy_from_enclave(val);
            user
        }
    }

    /// Creates an owned `User<T>` from a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `T`, is freeable with the `free`
    /// usercall and the alignment of `T`, and is uniquely owned.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        // SAFETY: the caller must uphold the safety contract for `from_raw`.
        unsafe { T::check_ptr(ptr) };
        User(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Converts this value into a raw pointer. The value will no longer be
    /// automatically freed.
    pub fn into_raw(self) -> *mut T {
        ManuallyDrop::new(self).0.as_ptr() as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> User<T>
where
    T: UserSafe,
{
    /// Allocates space for `T` in user memory.
    pub fn uninitialized() -> Self {
        Self::new_uninit_bytes(size_of::<T>())
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> User<[T]>
where
    [T]: UserSafe,
{
    /// Allocates space for a `[T]` of `n` elements in user memory.
    pub fn uninitialized(n: usize) -> Self {
        Self::new_uninit_bytes(n * size_of::<T>())
    }

    /// Creates an owned `User<[T]>` from a raw thin pointer and a slice length.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `len` elements of `T`, is
    /// freeable with the `free` usercall and the alignment of `T`, and is
    /// uniquely owned.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range does not fit in the address space
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        User(unsafe { <[T]>::from_raw_sized(ptr as _, len * size_of::<T>()) })
    }
}

/// Divide the slice `(ptr, len)` into three parts, where the middle part is
/// aligned to `u64`.
///
/// The return values `(prefix_len, mid_len, suffix_len)` add back up to `len`.
/// The return values are such that the memory region `(ptr + prefix_len,
/// mid_len)` is the largest possible region where `ptr + prefix_len` is aligned
/// to `u64` and `mid_len` is a multiple of the byte size of `u64`. This means
/// that `prefix_len` and `suffix_len` are guaranteed to be less than the byte
/// size of `u64`, and that `(ptr, prefix_len)` and `(ptr + prefix_len +
/// mid_len, suffix_len)` don't straddle an alignment boundary.
// Standard Rust functions such as `<[u8]>::align_to::<u64>` and
// `<*const u8>::align_offset` aren't _guaranteed_ to compute the largest
// possible middle region, and as such can't be used.
fn u64_align_to_guaranteed(ptr: *const u8, mut len: usize) -> (usize, usize, usize) {
    const QWORD_SIZE: usize = size_of::<u64>();

    let offset = ptr as usize % QWORD_SIZE;

    let prefix_len = if intrinsics::unlikely(offset > 0) { QWORD_SIZE - offset } else { 0 };

    len = match len.checked_sub(prefix_len) {
        Some(remaining_len) => remaining_len,
        None => return (len, 0, 0),
    };

    let suffix_len = len % QWORD_SIZE;
    len -= suffix_len;

    (prefix_len, len, suffix_len)
}

unsafe fn copy_quadwords(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        asm!(
            "rep movsq (%rsi), (%rdi)",
            inout("rcx") len / 8 => _,
            inout("rdi") dst => _,
            inout("rsi") src => _,
            options(att_syntax, nostack, preserves_flags)
        );
    }
}

/// Copies `len` bytes of data from enclave pointer `src` to userspace `dst`
///
/// This function mitigates stale data vulnerabilities by ensuring all writes to untrusted memory are either:
///  - preceded by the VERW instruction and followed by the MFENCE; LFENCE instruction sequence
///  - or are in multiples of 8 bytes, aligned to an 8-byte boundary
///
/// # Panics
/// This function panics if:
///
/// * The `src` pointer is null
/// * The `dst` pointer is null
/// * The `src` memory range is not in enclave memory
/// * The `dst` memory range is not in user memory
///
/// # References
///  - https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-00615.html
///  - https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/processor-mmio-stale-data-vulnerabilities.html#inpage-nav-3-2-2
pub(crate) unsafe fn copy_to_userspace(src: *const u8, dst: *mut u8, len: usize) {
    /// Like `ptr::copy(src, dst, len)`, except it uses the Intel-recommended
    /// instruction sequence for unaligned writes.
    unsafe fn write_bytewise_to_userspace(src: *const u8, dst: *mut u8, len: usize) {
        if intrinsics::likely(len == 0) {
            return;
        }

        unsafe {
            let mut seg_sel: u16 = 0;
            for off in 0..len {
                asm!("
                    mov %ds, ({seg_sel})
                    verw ({seg_sel})
                    movb {val}, ({dst})
                    mfence
                    lfence
                    ",
                    val = in(reg_byte) *src.add(off),
                    dst = in(reg) dst.add(off),
                    seg_sel = in(reg) &mut seg_sel,
                    options(nostack, att_syntax)
                );
            }
        }
    }

    assert!(!src.is_null());
    assert!(!dst.is_null());
    assert!(is_enclave_range(src, len));
    assert!(is_user_range(dst, len));
    assert!(len < isize::MAX as usize);
    assert!(!src.addr().overflowing_add(len).1);
    assert!(!dst.addr().overflowing_add(len).1);

    unsafe {
        let (len1, len2, len3) = u64_align_to_guaranteed(dst, len);
        let (src1, dst1) = (src, dst);
        let (src2, dst2) = (src1.add(len1), dst1.add(len1));
        let (src3, dst3) = (src2.add(len2), dst2.add(len2));

        write_bytewise_to_userspace(src1, dst1, len1);
        copy_quadwords(src2, dst2, len2);
        write_bytewise_to_userspace(src3, dst3, len3);
    }
}

/// Copies `len` bytes of data from userspace pointer `src` to enclave pointer `dst`
///
/// This function mitigates AEPIC leak vulnerabilities by ensuring all reads from untrusted memory are 8-byte aligned
///
/// # Panics
/// This function panics if:
///
/// * The `src` pointer is null
/// * The `dst` pointer is null
/// * The `src` memory range is not in user memory
/// * The `dst` memory range is not in enclave memory
///
/// # References
///  - https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-00657.html
///  - https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/advisory-guidance/stale-data-read-from-xapic.html
pub(crate) unsafe fn copy_from_userspace(src: *const u8, dst: *mut u8, len: usize) {
    /// Like `ptr::copy(src, dst, len)`, except it uses only u64-aligned reads.
    ///
    /// # Safety
    /// The source memory region must not straddle an alignment boundary.
    unsafe fn read_misaligned_from_userspace(src: *const u8, dst: *mut u8, len: usize) {
        if intrinsics::likely(len == 0) {
            return;
        }

        unsafe {
            let offset: usize;
            let data: u64;
            // doing a memory read that's potentially out of bounds for `src`,
            // this isn't supported by Rust, so have to use assembly
            asm!("
                movl {src:e}, {offset:e}
                andl $7, {offset:e}
                andq $-8, {src}
                movq ({src}), {dst}
                ",
                src = inout(reg) src => _,
                offset = out(reg) offset,
                dst = out(reg) data,
                options(nostack, att_syntax, readonly, pure)
            );
            let data = data.to_le_bytes();
            ptr::copy_nonoverlapping(data.as_ptr().add(offset), dst, len);
        }
    }

    assert!(!src.is_null());
    assert!(!dst.is_null());
    assert!(is_user_range(src, len));
    assert!(is_enclave_range(dst, len));
    assert!(len < isize::MAX as usize);
    assert!(!(src as usize).overflowing_add(len).1);
    assert!(!(dst as usize).overflowing_add(len).1);

    unsafe {
        let (len1, len2, len3) = u64_align_to_guaranteed(src, len);
        let (src1, dst1) = (src, dst);
        let (src2, dst2) = (src1.add(len1), dst1.add(len1));
        let (src3, dst3) = (src2.add(len2), dst2.add(len2));

        read_misaligned_from_userspace(src1, dst1, len1);
        copy_quadwords(src2, dst2, len2);
        read_misaligned_from_userspace(src3, dst3, len3);
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: ?Sized> UserRef<'a, T>
where
    T: UserSafe,
{
    /// Creates a `UserRef<'a, T>` from a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `T`.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_ptr(ptr: *const T) -> Self {
        // SAFETY: The caller must uphold the safety contract for `from_ptr`.
        unsafe {
            T::check_ptr(ptr);
            Self::from_non_null_unchecked(NonNull::new_unchecked(ptr as _))
        }
    }

    /// Creates a `UserRef<'a, T>` from a non-null pointer without performing further checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// * `non_null` points to `T`
    /// * The pointer is aligned
    /// * The pointed-to range is in user memory
    unsafe fn from_non_null_unchecked(non_null: NonNull<T>) -> Self {
        UserRef { data: non_null, _phantom: PhantomData }
    }

    /// Copies the value from user memory and place it into `dest`.
    ///
    /// # Panics
    /// This function panics if the destination doesn't have the same size as
    /// the source. This can happen for dynamically-sized types such as slices.
    pub fn copy_to_enclave<U: ?Sized + UserSafeCopyDestination<T>>(self, dest: &mut U) {
        unsafe {
            assert_eq!(size_of_val(dest), mem::size_of_val_raw(self.data.as_ptr()));
            copy_from_userspace(
                self.as_raw_ptr() as *const u8,
                dest.as_mut_ptr() as *mut u8,
                size_of_val(dest),
            );
        }
    }

    /// Obtain a raw pointer from this reference.
    pub fn as_raw_ptr(self) -> *const T {
        self.data.as_ptr()
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T> UserRef<'a, T>
where
    T: UserSafe,
{
    /// Copies the value from user memory into enclave memory.
    pub fn to_enclave(self) -> T {
        unsafe {
            let mut data = mem::MaybeUninit::uninit();
            copy_from_userspace(self.as_raw_ptr() as _, data.as_mut_ptr() as _, size_of::<T>());
            data.assume_init()
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: ?Sized> UserMut<'a, T>
where
    T: UserSafe,
{
    /// Creates a `UserMut<'a, [T]>` from a raw pointer. See the struct
    /// documentation for the nuances regarding a `UserMut<'a, T>`.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `T`.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_mut_ptr(ptr: *mut T) -> Self {
        // SAFETY: The caller must uphold the safety contract for `from_mut_ptr`.
        unsafe {
            T::check_ptr(ptr);
            UserMut::from_non_null_unchecked(unsafe { NonNull::new_unchecked(ptr) })
        }
    }

    /// Creates a `UserMut<'a, T>` from a non-null pointer without performing further checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// * `non_null` points to `T`
    /// * The pointer is aligned
    /// * The pointed-to range is in user memory
    unsafe fn from_non_null_unchecked(non_null: NonNull<T>) -> Self {
        UserMut { data: non_null, _phantom: PhantomData }
    }

    /// Copies `val` into user memory.
    ///
    /// # Panics
    /// This function panics if the destination doesn't have the same size as
    /// the source. This can happen for dynamically-sized types such as slices.
    pub fn copy_from_enclave(self, val: &T) {
        unsafe {
            assert_eq!(size_of_val(val), mem::size_of_val_raw(self.data.as_ptr()));
            copy_to_userspace(
                val as *const T as *const u8,
                self.data.as_ptr() as *mut u8,
                size_of_val(val),
            );
        }
    }

    /// Obtain a raw pointer from this reference.
    pub fn as_raw_mut_ptr(self) -> *mut T {
        self.data.as_ptr()
    }

    /// Limited convenience implementation of reborrowing that ties return type lifetime to lifetime of borrow of `self`.
    // FIXME: remove when `Reborrow` trait ergonomics improve
    pub fn reborrow(&mut self) -> UserMut<'_, T> {
        unsafe { Self::from_non_null_unchecked(self.data) }
    }

    /// Limited convenience implementation of owned dereferencing that ties return type lifetime to lifetime of borrow of self.
    // FIXME: remove when `CoerceShared` trait ergonomics improve
    pub fn coerce_shared(&self) -> UserRef<'_, T> {
        unsafe { UserRef::from_non_null_unchecked(self.data) }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T> UserRef<'a, [T]>
where
    [T]: UserSafe,
{
    /// Creates a `UserRef<'a, [T]>` from a raw thin pointer and a slice length.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `n` elements of `T`.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range does not fit in the address space
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        // SAFETY: The caller must uphold the safety contract for `from_raw_parts`.
        unsafe {
            let data = <[T]>::from_raw_sized(ptr as _, len * size_of::<T>());
            UserRef::from_non_null_unchecked(data)
        }
    }

    /// Obtain a raw pointer to the first element of this user slice.
    pub fn as_ptr(self) -> *const T {
        self.data.as_ptr() as _
    }

    /// Obtain the number of elements in this user slice.
    pub fn len(self) -> usize {
        unsafe { self.data.as_ptr().len() }
    }

    /// Copies the value from user memory and appends it to `dest`.
    pub fn append_to_enclave_vec(self, dest: &mut Vec<T>) {
        dest.reserve(self.len());
        self.copy_to_enclave(&mut dest.spare_capacity_mut()[..self.len()]);
        // SAFETY: We reserve enough space above.
        unsafe { dest.set_len(dest.len() + self.len()) };
    }

    /// Copies the value from user memory into a vector in enclave memory.
    pub fn to_enclave(self) -> Vec<T> {
        let mut ret = Vec::with_capacity(self.len());
        self.append_to_enclave_vec(&mut ret);
        ret
    }

    /// Returns an iterator over the slice.
    pub fn iter(self) -> Iter<'a, T> {
        Iter(self)
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T> UserMut<'a, [T]>
where
    [T]: UserSafe,
{
    /// Creates a `UserMut<'a, [T]>` from a raw thin pointer and a slice length.
    /// See the struct documentation for the nuances regarding `UserMut<'a, T>`.
    ///
    /// # Safety
    /// The caller must ensure `ptr` points to `n` elements of `T`.
    ///
    /// # Panics
    /// This function panics if:
    ///
    /// * The pointer is not aligned
    /// * The pointer is null
    /// * The pointed-to range does not fit in the address space
    /// * The pointed-to range is not in user memory
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> Self {
        // SAFETY: The caller must uphold the safety contract for `from_raw_parts_mut`.
        unsafe {
            let data = <[T]>::from_raw_sized(ptr as _, len * size_of::<T>());
            UserMut::from_non_null_unchecked(data)
        }
    }

    /// Obtain a raw pointer to the first element of this user slice.
    pub fn as_mut_ptr(self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(self) -> IterMut<'a, T> {
        IterMut(self)
    }
}

/// Immutable user slice iterator
///
/// This struct is created by the [`iter`] method on `UserRef<[T]>`.
///
/// [`iter`]: UserRef::iter
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Iter<'a, T: 'a>(UserRef<'a, [T]>)
where
    [T]: UserSafe;

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe> Iterator for Iter<'a, T>
where
    [T]: UserSafe,
{
    type Item = UserRef<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let cur_user_ref = self.0.data;

        if cur_user_ref.len() == 0 {
            return None;
        }

        // SAFETY; `cur_user_ref` was derived from a valid [UserRef] and has length >= 1
        unsafe {
            let item = UserRef::from_non_null_unchecked(cur_user_ref.as_non_null_ptr());
            self.0.data = cur_user_ref.get_unchecked_mut(1..);
            Some(item)
        }
    }
}

/// Mutable user slice iterator
///
/// This struct is created by the [`iter_mut`] method on `UserMut<[T]>`.
///
/// [`iter_mut`]: UserMut::iter_mut
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct IterMut<'a, T: 'a>(UserMut<'a, [T]>)
where
    [T]: UserSafe;

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe> Iterator for IterMut<'a, T>
where
    [T]: UserSafe,
{
    type Item = UserMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let cur_user_mut = self.0.data;

        if cur_user_mut.len() == 0 {
            return None;
        }

        // SAFETY; `cur_user_mut` was derived from a valid [UserMut] and has length >= 1
        unsafe {
            let item = UserMut::from_non_null_unchecked(cur_user_mut.as_non_null_ptr());
            self.0.data = cur_user_mut.get_unchecked_mut(1..);
            Some(item)
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized + UserSafe> User<T> {
    /// Equivalent of `& *` on [Box]; create an immutable user reference to the memory that `Self` points to
    pub fn as_user_ref(&self) -> UserRef<'_, T> {
        unsafe { UserRef::from_non_null_unchecked(self.0) }
    }

    /// Equivalent of `&mut *` on [Box]; create a mutable user reference to the memory that `Self` points to
    pub fn as_user_mut(&mut self) -> UserMut<'_, T> {
        unsafe { UserMut::from_non_null_unchecked(self.0) }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> Drop for User<T>
where
    T: UserSafe,
{
    fn drop(&mut self) {
        unsafe {
            let ptr = self.0.as_ptr();
            super::free(ptr as _, mem::size_of_val_raw(ptr), T::align_of());
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe + Unsize<U> + ?Sized, U: UserSafe + ?Sized> CoerceUnsized<UserRef<'a, U>>
    for UserRef<'a, T>
{
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe + Unsize<U> + ?Sized, U: UserSafe + ?Sized> CoerceUnsized<UserMut<'a, U>>
    for UserMut<'a, T>
{
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T> UserRef<'a, [T]>
where
    [T]: UserSafe,
{
    /// Index this user reference.
    ///
    /// Implemented as an instance method rather than an [std::ops::Index] trait instance, since the
    /// returned [UserRef] is owned.
    #[inline]
    pub fn index<I>(self, index: I) -> UserRef<'a, I::Output>
    where
        I: SliceIndex<[T]>,
        I::Output: UserSafe,
    {
        unsafe {
            if let Some(slice) = index.get_raw(self.as_raw_ptr()) {
                UserRef::from_ptr(slice)
            } else {
                rtabort!("index out of range for user slice");
            }
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T> UserMut<'a, [T]>
where
    [T]: UserSafe,
{
    /// Mutably index this user reference.
    ///
    /// Implemented as an instance method rather than an [std::ops::IndexMut] trait instance, since the
    /// returned [UserMut] is owned.
    #[inline]
    pub fn index_mut<'b, I>(self, index: I) -> UserMut<'a, I::Output>
    where
        I: SliceIndex<[T]>,
        I::Output: UserSafe,
    {
        unsafe {
            if let Some(slice) = index.get_raw_mut(self.as_raw_mut_ptr()) {
                UserMut::from_mut_ptr(slice)
            } else {
                rtabort!("index out of range for user slice");
            }
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a> UserRef<'a, super::raw::ByteBuffer> {
    /// Copies the user memory range pointed to by the user `ByteBuffer` to
    /// enclave memory.
    ///
    /// # Panics
    /// This function panics if, in the user `ByteBuffer`:
    ///
    /// * The pointer is null
    /// * The pointed-to range does not fit in the address space
    /// * The pointed-to range is not in user memory
    pub fn copy_user_buffer(self) -> Vec<u8> {
        unsafe {
            let buf = self.to_enclave();
            if buf.len > 0 {
                User::from_raw_parts(buf.data as _, buf.len).as_user_ref().to_enclave()
            } else {
                // Mustn't look at `data` or call `free` if `len` is `0`.
                Vec::with_capacity(0)
            }
        }
    }
}
