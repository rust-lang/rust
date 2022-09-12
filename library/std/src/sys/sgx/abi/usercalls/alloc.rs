#![allow(unused)]

use crate::arch::asm;
use crate::cell::UnsafeCell;
use crate::cmp;
use crate::convert::TryInto;
use crate::mem;
use crate::ops::{CoerceUnsized, Deref, DerefMut, Index, IndexMut};
use crate::ptr::{self, NonNull};
use crate::slice;
use crate::slice::SliceIndex;

use super::super::mem::{is_enclave_range, is_user_range};
use fortanix_sgx_abi::*;

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
    /// Equivalent to `mem::align_of::<Self>`.
    fn align_of() -> usize;

    /// Construct a pointer to `Self` given a memory range in user space.
    ///
    /// N.B., this takes a size, not a length!
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory range is in user memory, is the
    /// correct size and is correctly aligned and points to the right type.
    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self;

    /// Construct a pointer to `Self` given a memory range.
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
        assert!(is_user_range(ptr as _, mem::size_of_val(unsafe { &*ptr })));
        assert!(!ptr.is_null());
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafeSized> UserSafe for T {
    fn align_of() -> usize {
        mem::align_of::<T>()
    }

    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self {
        assert_eq!(size, mem::size_of::<T>());
        ptr as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
unsafe impl<T: UserSafeSized> UserSafe for [T] {
    fn align_of() -> usize {
        mem::align_of::<T>()
    }

    /// # Safety
    /// Behavior is undefined if any of these conditions are violated:
    /// * `ptr` must be [valid] for writes of `size` many bytes, and it must be
    ///   properly aligned.
    ///
    /// [valid]: core::ptr#safety
    /// # Panics
    ///
    /// This function panics if:
    ///
    /// * the element size is not a factor of the size
    unsafe fn from_raw_sized_unchecked(ptr: *mut u8, size: usize) -> *mut Self {
        let elem_size = mem::size_of::<T>();
        assert_eq!(size % elem_size, 0);
        let len = size / elem_size;
        // SAFETY: The caller must uphold the safety contract for `from_raw_sized_unchecked`
        unsafe { slice::from_raw_parts_mut(ptr as _, len) }
    }
}

/// A reference to some type in userspace memory. `&UserRef<T>` is equivalent
/// to `&T` in enclave memory. Access to the memory is only allowed by copying
/// to avoid TOCTTOU issues. After copying, code should make sure to completely
/// check the value before use.
///
/// It is also possible to obtain a mutable reference `&mut UserRef<T>`. Unlike
/// regular mutable references, these are not exclusive. Userspace may always
/// write to the backing memory at any time, so it can't be assumed that there
/// the pointed-to memory is uniquely borrowed. The two different reference types
/// are used solely to indicate intent: a mutable reference is for writing to
/// user memory, an immutable reference for reading from user memory.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct UserRef<T: ?Sized>(UnsafeCell<T>);
/// An owned type in userspace memory. `User<T>` is equivalent to `Box<T>` in
/// enclave memory. Access to the memory is only allowed by copying to avoid
/// TOCTTOU issues. The user memory will be freed when the value is dropped.
/// After copying, code should make sure to completely check the value before
/// use.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct User<T: UserSafe + ?Sized>(NonNull<UserRef<T>>);

trait NewUserRef<T: ?Sized> {
    unsafe fn new_userref(v: T) -> Self;
}

impl<T: ?Sized> NewUserRef<*mut T> for NonNull<UserRef<T>> {
    unsafe fn new_userref(v: *mut T) -> Self {
        // SAFETY: The caller has guaranteed the pointer is valid
        unsafe { NonNull::new_unchecked(v as _) }
    }
}

impl<T: ?Sized> NewUserRef<NonNull<T>> for NonNull<UserRef<T>> {
    unsafe fn new_userref(v: NonNull<T>) -> Self {
        // SAFETY: The caller has guaranteed the pointer is valid
        unsafe { NonNull::new_userref(v.as_ptr()) }
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
                User(NonNull::new_userref(v))
            } else {
                rtabort!("Got invalid pointer from alloc() usercall")
            }
        }
    }

    /// Copies `val` into freshly allocated space in user memory.
    pub fn new_from_enclave(val: &T) -> Self {
        unsafe {
            let mut user = Self::new_uninit_bytes(mem::size_of_val(val));
            user.copy_from_enclave(val);
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
        User(unsafe { NonNull::new_userref(ptr) })
    }

    /// Converts this value into a raw pointer. The value will no longer be
    /// automatically freed.
    pub fn into_raw(self) -> *mut T {
        let ret = self.0;
        mem::forget(self);
        ret.as_ptr() as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> User<T>
where
    T: UserSafe,
{
    /// Allocate space for `T` in user memory.
    pub fn uninitialized() -> Self {
        Self::new_uninit_bytes(mem::size_of::<T>())
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> User<[T]>
where
    [T]: UserSafe,
{
    /// Allocate space for a `[T]` of `n` elements in user memory.
    pub fn uninitialized(n: usize) -> Self {
        Self::new_uninit_bytes(n * mem::size_of::<T>())
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
        User(unsafe {
            NonNull::new_userref(<[T]>::from_raw_sized(ptr as _, len * mem::size_of::<T>()))
        })
    }
}

// Split a memory region ptr..ptr + len into three parts:
//   +--------+
//   | small0 | Chunk smaller than 8 bytes
//   +--------+
//   |   big  | Chunk 8-byte aligned, and size a multiple of 8 bytes
//   +--------+
//   | small1 | Chunk smaller than 8 bytes
//   +--------+
fn region_as_aligned_chunks(ptr: *const u8, len: usize) -> (usize, usize, usize) {
    let small0_size = if ptr as usize % 8 == 0 { 0 } else { 8 - ptr as usize % 8 };
    let small1_size = (len - small0_size as usize) % 8;
    let big_size = len - small0_size as usize - small1_size as usize;

    (small0_size, big_size, small1_size)
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
    unsafe fn copy_bytewise_to_userspace(src: *const u8, dst: *mut u8, len: usize) {
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
                    val = in(reg_byte) *src.offset(off as isize),
                    dst = in(reg) dst.offset(off as isize),
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
    assert!(!(src as usize).overflowing_add(len).1);
    assert!(!(dst as usize).overflowing_add(len).1);

    if len < 8 {
        // Can't align on 8 byte boundary: copy safely byte per byte
        unsafe {
            copy_bytewise_to_userspace(src, dst, len);
        }
    } else if len % 8 == 0 && dst.is_aligned_to(8) {
        // Copying 8-byte aligned quadwords: copy quad word per quad word
        unsafe {
            copy_quadwords(src, dst, len);
        }
    } else {
        // Split copies into three parts:
        //   +--------+
        //   | small0 | Chunk smaller than 8 bytes
        //   +--------+
        //   |   big  | Chunk 8-byte aligned, and size a multiple of 8 bytes
        //   +--------+
        //   | small1 | Chunk smaller than 8 bytes
        //   +--------+
        let (small0_size, big_size, small1_size) = region_as_aligned_chunks(dst, len);

        unsafe {
            // Copy small0
            copy_bytewise_to_userspace(src, dst, small0_size as _);

            // Copy big
            let big_src = src.offset(small0_size as _);
            let big_dst = dst.offset(small0_size as _);
            copy_quadwords(big_src as _, big_dst, big_size);

            // Copy small1
            let small1_src = src.offset(big_size as isize + small0_size as isize);
            let small1_dst = dst.offset(big_size as isize + small0_size as isize);
            copy_bytewise_to_userspace(small1_src, small1_dst, small1_size as _);
        }
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
    // Copies memory region `src..src + len` to the enclave at `dst`. The source memory region
    // is:
    //  - strictly less than 8 bytes in size and may be
    //  - located at a misaligned memory location
    fn copy_misaligned_chunk_to_enclave(src: *const u8, dst: *mut u8, len: usize) {
        let mut tmp_buff = [0u8; 16];

        unsafe {
            // Compute an aligned memory region to read from
            // +--------+ <-- aligned_src + aligned_len (8B-aligned)
            // |  pad1  |
            // +--------+ <-- src + len (misaligned)
            // |        |
            // |        |
            // |        |
            // +--------+ <-- src (misaligned)
            // |  pad0  |
            // +--------+ <-- aligned_src (8B-aligned)
            let pad0_size = src as usize % 8;
            let aligned_src = src.sub(pad0_size);

            let pad1_size = 8 - (src.add(len) as usize % 8);
            let aligned_len = pad0_size + len + pad1_size;

            debug_assert!(len < 8);
            debug_assert_eq!(aligned_src as usize % 8, 0);
            debug_assert_eq!(aligned_len % 8, 0);
            debug_assert!(aligned_len <= 16);

            // Copy the aligned buffer to a temporary buffer
            // Note: copying from a slightly different memory location is a bit odd. In this case it
            // can't lead to page faults or inadvertent copying from the enclave as we only ensured
            // that the `src` pointer is aligned at an 8 byte boundary. As pages are 4096 bytes
            // aligned, `aligned_src` must be on the same page as `src`. A similar argument can be made
            // for `src + len`
            copy_quadwords(aligned_src as _, tmp_buff.as_mut_ptr(), aligned_len);

            // Copy the correct parts of the temporary buffer to the destination
            ptr::copy(tmp_buff.as_ptr().add(pad0_size), dst, len);
        }
    }

    assert!(!src.is_null());
    assert!(!dst.is_null());
    assert!(is_user_range(src, len));
    assert!(is_enclave_range(dst, len));
    assert!(!(src as usize).overflowing_add(len + 8).1);
    assert!(!(dst as usize).overflowing_add(len + 8).1);

    if len < 8 {
        copy_misaligned_chunk_to_enclave(src, dst, len);
    } else if len % 8 == 0 && src as usize % 8 == 0 {
        // Copying 8-byte aligned quadwords: copy quad word per quad word
        unsafe {
            copy_quadwords(src, dst, len);
        }
    } else {
        // Split copies into three parts:
        //   +--------+
        //   | small0 | Chunk smaller than 8 bytes
        //   +--------+
        //   |   big  | Chunk 8-byte aligned, and size a multiple of 8 bytes
        //   +--------+
        //   | small1 | Chunk smaller than 8 bytes
        //   +--------+
        let (small0_size, big_size, small1_size) = region_as_aligned_chunks(dst, len);

        unsafe {
            // Copy small0
            copy_misaligned_chunk_to_enclave(src, dst, small0_size);

            // Copy big
            let big_src = src.add(small0_size);
            let big_dst = dst.add(small0_size);
            copy_quadwords(big_src, big_dst, big_size);

            // Copy small1
            let small1_src = src.add(big_size + small0_size);
            let small1_dst = dst.add(big_size + small0_size);
            copy_misaligned_chunk_to_enclave(small1_src, small1_dst, small1_size);
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> UserRef<T>
where
    T: UserSafe,
{
    /// Creates a `&UserRef<[T]>` from a raw pointer.
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
    pub unsafe fn from_ptr<'a>(ptr: *const T) -> &'a Self {
        // SAFETY: The caller must uphold the safety contract for `from_ptr`.
        unsafe { T::check_ptr(ptr) };
        unsafe { &*(ptr as *const Self) }
    }

    /// Creates a `&mut UserRef<[T]>` from a raw pointer. See the struct
    /// documentation for the nuances regarding a `&mut UserRef<T>`.
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
    pub unsafe fn from_mut_ptr<'a>(ptr: *mut T) -> &'a mut Self {
        // SAFETY: The caller must uphold the safety contract for `from_mut_ptr`.
        unsafe { T::check_ptr(ptr) };
        unsafe { &mut *(ptr as *mut Self) }
    }

    /// Copies `val` into user memory.
    ///
    /// # Panics
    /// This function panics if the destination doesn't have the same size as
    /// the source. This can happen for dynamically-sized types such as slices.
    pub fn copy_from_enclave(&mut self, val: &T) {
        unsafe {
            assert_eq!(mem::size_of_val(val), mem::size_of_val(&*self.0.get()));
            copy_to_userspace(
                val as *const T as *const u8,
                self.0.get() as *mut T as *mut u8,
                mem::size_of_val(val),
            );
        }
    }

    /// Copies the value from user memory and place it into `dest`.
    ///
    /// # Panics
    /// This function panics if the destination doesn't have the same size as
    /// the source. This can happen for dynamically-sized types such as slices.
    pub fn copy_to_enclave(&self, dest: &mut T) {
        unsafe {
            assert_eq!(mem::size_of_val(dest), mem::size_of_val(&*self.0.get()));
            copy_from_userspace(
                self.0.get() as *const T as *const u8,
                dest as *mut T as *mut u8,
                mem::size_of_val(dest),
            );
        }
    }

    /// Obtain a raw pointer from this reference.
    pub fn as_raw_ptr(&self) -> *const T {
        self as *const _ as _
    }

    /// Obtain a raw pointer from this reference.
    pub fn as_raw_mut_ptr(&mut self) -> *mut T {
        self as *mut _ as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> UserRef<T>
where
    T: UserSafe,
{
    /// Copies the value from user memory into enclave memory.
    pub fn to_enclave(&self) -> T {
        unsafe {
            let mut data: T = mem::MaybeUninit::uninit().assume_init();
            copy_from_userspace(self.0.get() as _, &mut data as *mut T as _, mem::size_of::<T>());
            data
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T> UserRef<[T]>
where
    [T]: UserSafe,
{
    /// Creates a `&UserRef<[T]>` from a raw thin pointer and a slice length.
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
    pub unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a Self {
        // SAFETY: The caller must uphold the safety contract for `from_raw_parts`.
        unsafe {
            &*(<[T]>::from_raw_sized(ptr as _, len * mem::size_of::<T>()).as_ptr() as *const Self)
        }
    }

    /// Creates a `&mut UserRef<[T]>` from a raw thin pointer and a slice length.
    /// See the struct documentation for the nuances regarding a
    /// `&mut UserRef<T>`.
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
    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut Self {
        // SAFETY: The caller must uphold the safety contract for `from_raw_parts_mut`.
        unsafe {
            &mut *(<[T]>::from_raw_sized(ptr as _, len * mem::size_of::<T>()).as_ptr() as *mut Self)
        }
    }

    /// Obtain a raw pointer to the first element of this user slice.
    pub fn as_ptr(&self) -> *const T {
        self.0.get() as _
    }

    /// Obtain a raw pointer to the first element of this user slice.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.get() as _
    }

    /// Obtain the number of elements in this user slice.
    pub fn len(&self) -> usize {
        unsafe { (*self.0.get()).len() }
    }

    /// Copies the value from user memory and place it into `dest`. Afterwards,
    /// `dest` will contain exactly `self.len()` elements.
    ///
    /// # Panics
    /// This function panics if the destination doesn't have the same size as
    /// the source. This can happen for dynamically-sized types such as slices.
    pub fn copy_to_enclave_vec(&self, dest: &mut Vec<T>) {
        if let Some(missing) = self.len().checked_sub(dest.capacity()) {
            dest.reserve(missing)
        }
        // SAFETY: We reserve enough space above.
        unsafe { dest.set_len(self.len()) };
        self.copy_to_enclave(&mut dest[..]);
    }

    /// Copies the value from user memory into a vector in enclave memory.
    pub fn to_enclave(&self) -> Vec<T> {
        let mut ret = Vec::with_capacity(self.len());
        self.copy_to_enclave_vec(&mut ret);
        ret
    }

    /// Returns an iterator over the slice.
    pub fn iter(&self) -> Iter<'_, T>
    where
        T: UserSafe, // FIXME: should be implied by [T]: UserSafe?
    {
        unsafe { Iter((&*self.as_raw_ptr()).iter()) }
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> IterMut<'_, T>
    where
        T: UserSafe, // FIXME: should be implied by [T]: UserSafe?
    {
        unsafe { IterMut((&mut *self.as_raw_mut_ptr()).iter_mut()) }
    }
}

/// Immutable user slice iterator
///
/// This struct is created by the `iter` method on `UserRef<[T]>`.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct Iter<'a, T: 'a + UserSafe>(slice::Iter<'a, T>);

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe> Iterator for Iter<'a, T> {
    type Item = &'a UserRef<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next().map(|e| UserRef::from_ptr(e)) }
    }
}

/// Mutable user slice iterator
///
/// This struct is created by the `iter_mut` method on `UserRef<[T]>`.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub struct IterMut<'a, T: 'a + UserSafe>(slice::IterMut<'a, T>);

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<'a, T: UserSafe> Iterator for IterMut<'a, T> {
    type Item = &'a mut UserRef<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next().map(|e| UserRef::from_mut_ptr(e)) }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> Deref for User<T>
where
    T: UserSafe,
{
    type Target = UserRef<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0.as_ptr() }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> DerefMut for User<T>
where
    T: UserSafe,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0.as_ptr() }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: ?Sized> Drop for User<T>
where
    T: UserSafe,
{
    fn drop(&mut self) {
        unsafe {
            let ptr = (*self.0.as_ptr()).0.get();
            super::free(ptr as _, mem::size_of_val(&mut *ptr), T::align_of());
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: CoerceUnsized<U>, U> CoerceUnsized<UserRef<U>> for UserRef<T> {}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T, I> Index<I> for UserRef<[T]>
where
    [T]: UserSafe,
    I: SliceIndex<[T]>,
    I::Output: UserSafe,
{
    type Output = UserRef<I::Output>;

    #[inline]
    fn index(&self, index: I) -> &UserRef<I::Output> {
        unsafe {
            if let Some(slice) = index.get(&*self.as_raw_ptr()) {
                UserRef::from_ptr(slice)
            } else {
                rtabort!("index out of range for user slice");
            }
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T, I> IndexMut<I> for UserRef<[T]>
where
    [T]: UserSafe,
    I: SliceIndex<[T]>,
    I::Output: UserSafe,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut UserRef<I::Output> {
        unsafe {
            if let Some(slice) = index.get_mut(&mut *self.as_raw_mut_ptr()) {
                UserRef::from_mut_ptr(slice)
            } else {
                rtabort!("index out of range for user slice");
            }
        }
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl UserRef<super::raw::ByteBuffer> {
    /// Copies the user memory range pointed to by the user `ByteBuffer` to
    /// enclave memory.
    ///
    /// # Panics
    /// This function panics if, in the user `ByteBuffer`:
    ///
    /// * The pointer is null
    /// * The pointed-to range does not fit in the address space
    /// * The pointed-to range is not in user memory
    pub fn copy_user_buffer(&self) -> Vec<u8> {
        unsafe {
            let buf = self.to_enclave();
            if buf.len > 0 {
                User::from_raw_parts(buf.data as _, buf.len).to_enclave()
            } else {
                // Mustn't look at `data` or call `free` if `len` is `0`.
                Vec::with_capacity(0)
            }
        }
    }
}
