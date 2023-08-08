use core::cell::RefCell;
use core::mem::{self, MaybeUninit};
use core::num::NonZeroUsize;
use core::ptr;
use core::ptr::*;
use std::fmt::{Debug, Display};

#[test]
fn test_const_from_raw_parts() {
    const SLICE: &[u8] = &[1, 2, 3, 4];
    const FROM_RAW: &[u8] = unsafe { &*slice_from_raw_parts(SLICE.as_ptr(), SLICE.len()) };
    assert_eq!(SLICE, FROM_RAW);

    let slice = &[1, 2, 3, 4, 5];
    let from_raw = unsafe { &*slice_from_raw_parts(slice.as_ptr(), 2) };
    assert_eq!(&slice[..2], from_raw);
}

#[test]
fn test() {
    unsafe {
        #[repr(C)]
        struct Pair {
            fst: isize,
            snd: isize,
        }
        let mut p = Pair { fst: 10, snd: 20 };
        let pptr: *mut Pair = addr_of_mut!(p);
        let iptr: *mut isize = pptr as *mut isize;
        assert_eq!(*iptr, 10);
        *iptr = 30;
        assert_eq!(*iptr, 30);
        assert_eq!(p.fst, 30);

        *pptr = Pair { fst: 50, snd: 60 };
        assert_eq!(*iptr, 50);
        assert_eq!(p.fst, 50);
        assert_eq!(p.snd, 60);

        let v0 = vec![32000u16, 32001u16, 32002u16];
        let mut v1 = vec![0u16, 0u16, 0u16];

        copy(v0.as_ptr().offset(1), v1.as_mut_ptr().offset(1), 1);
        assert!((v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16));
        copy(v0.as_ptr().offset(2), v1.as_mut_ptr(), 1);
        assert!((v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 0u16));
        copy(v0.as_ptr(), v1.as_mut_ptr().offset(2), 1);
        assert!((v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 32000u16));
    }
}

#[test]
fn test_is_null() {
    let p: *const isize = null();
    assert!(p.is_null());

    let q = p.wrapping_offset(1);
    assert!(!q.is_null());

    let mp: *mut isize = null_mut();
    assert!(mp.is_null());

    let mq = mp.wrapping_offset(1);
    assert!(!mq.is_null());

    // Pointers to unsized types -- slices
    let s: &mut [u8] = &mut [1, 2, 3];
    let cs: *const [u8] = s;
    assert!(!cs.is_null());

    let ms: *mut [u8] = s;
    assert!(!ms.is_null());

    let cz: *const [u8] = &[];
    assert!(!cz.is_null());

    let mz: *mut [u8] = &mut [];
    assert!(!mz.is_null());

    let ncs: *const [u8] = null::<[u8; 3]>();
    assert!(ncs.is_null());

    let nms: *mut [u8] = null_mut::<[u8; 3]>();
    assert!(nms.is_null());

    // Pointers to unsized types -- trait objects
    let ci: *const dyn ToString = &3;
    assert!(!ci.is_null());

    let mi: *mut dyn ToString = &mut 3;
    assert!(!mi.is_null());

    let nci: *const dyn ToString = null::<isize>();
    assert!(nci.is_null());

    let nmi: *mut dyn ToString = null_mut::<isize>();
    assert!(nmi.is_null());

    extern "C" {
        type Extern;
    }
    let ec: *const Extern = null::<Extern>();
    assert!(ec.is_null());

    let em: *mut Extern = null_mut::<Extern>();
    assert!(em.is_null());
}

#[test]
fn test_as_ref() {
    unsafe {
        let p: *const isize = null();
        assert_eq!(p.as_ref(), None);

        let q: *const isize = &2;
        assert_eq!(q.as_ref().unwrap(), &2);

        let p: *mut isize = null_mut();
        assert_eq!(p.as_ref(), None);

        let q: *mut isize = &mut 2;
        assert_eq!(q.as_ref().unwrap(), &2);

        // Lifetime inference
        let u = 2isize;
        {
            let p = &u as *const isize;
            assert_eq!(p.as_ref().unwrap(), &2);
        }

        // Pointers to unsized types -- slices
        let s: &mut [u8] = &mut [1, 2, 3];
        let cs: *const [u8] = s;
        assert_eq!(cs.as_ref(), Some(&*s));

        let ms: *mut [u8] = s;
        assert_eq!(ms.as_ref(), Some(&*s));

        let cz: *const [u8] = &[];
        assert_eq!(cz.as_ref(), Some(&[][..]));

        let mz: *mut [u8] = &mut [];
        assert_eq!(mz.as_ref(), Some(&[][..]));

        let ncs: *const [u8] = null::<[u8; 3]>();
        assert_eq!(ncs.as_ref(), None);

        let nms: *mut [u8] = null_mut::<[u8; 3]>();
        assert_eq!(nms.as_ref(), None);

        // Pointers to unsized types -- trait objects
        let ci: *const dyn ToString = &3;
        assert!(ci.as_ref().is_some());

        let mi: *mut dyn ToString = &mut 3;
        assert!(mi.as_ref().is_some());

        let nci: *const dyn ToString = null::<isize>();
        assert!(nci.as_ref().is_none());

        let nmi: *mut dyn ToString = null_mut::<isize>();
        assert!(nmi.as_ref().is_none());
    }
}

#[test]
fn test_as_mut() {
    unsafe {
        let p: *mut isize = null_mut();
        assert!(p.as_mut() == None);

        let q: *mut isize = &mut 2;
        assert!(q.as_mut().unwrap() == &mut 2);

        // Lifetime inference
        let mut u = 2isize;
        {
            let p = &mut u as *mut isize;
            assert!(p.as_mut().unwrap() == &mut 2);
        }

        // Pointers to unsized types -- slices
        let s: &mut [u8] = &mut [1, 2, 3];
        let ms: *mut [u8] = s;
        assert_eq!(ms.as_mut(), Some(&mut [1, 2, 3][..]));

        let mz: *mut [u8] = &mut [];
        assert_eq!(mz.as_mut(), Some(&mut [][..]));

        let nms: *mut [u8] = null_mut::<[u8; 3]>();
        assert_eq!(nms.as_mut(), None);

        // Pointers to unsized types -- trait objects
        let mi: *mut dyn ToString = &mut 3;
        assert!(mi.as_mut().is_some());

        let nmi: *mut dyn ToString = null_mut::<isize>();
        assert!(nmi.as_mut().is_none());
    }
}

#[test]
fn test_ptr_addition() {
    unsafe {
        let xs = vec![5; 16];
        let mut ptr = xs.as_ptr();
        let end = ptr.offset(16);

        while ptr < end {
            assert_eq!(*ptr, 5);
            ptr = ptr.offset(1);
        }

        let mut xs_mut = xs;
        let mut m_ptr = xs_mut.as_mut_ptr();
        let m_end = m_ptr.offset(16);

        while m_ptr < m_end {
            *m_ptr += 5;
            m_ptr = m_ptr.offset(1);
        }

        assert!(xs_mut == vec![10; 16]);
    }
}

#[test]
fn test_ptr_subtraction() {
    unsafe {
        let xs = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut idx = 9;
        let ptr = xs.as_ptr();

        while idx >= 0 {
            assert_eq!(*(ptr.offset(idx as isize)), idx as isize);
            idx = idx - 1;
        }

        let mut xs_mut = xs;
        let m_start = xs_mut.as_mut_ptr();
        let mut m_ptr = m_start.offset(9);

        loop {
            *m_ptr += *m_ptr;
            if m_ptr == m_start {
                break;
            }
            m_ptr = m_ptr.offset(-1);
        }

        assert_eq!(xs_mut, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }
}

#[test]
fn test_set_memory() {
    let mut xs = [0u8; 20];
    let ptr = xs.as_mut_ptr();
    unsafe {
        write_bytes(ptr, 5u8, xs.len());
    }
    assert!(xs == [5u8; 20]);
}

#[test]
fn test_set_memory_const() {
    const XS: [u8; 20] = {
        let mut xs = [0u8; 20];
        let ptr = xs.as_mut_ptr();
        unsafe {
            ptr.write_bytes(5u8, xs.len());
        }
        xs
    };

    assert!(XS == [5u8; 20]);
}

#[test]
fn test_unsized_nonnull() {
    let xs: &[i32] = &[1, 2, 3];
    let ptr = unsafe { NonNull::new_unchecked(xs as *const [i32] as *mut [i32]) };
    let ys = unsafe { ptr.as_ref() };
    let zs: &[i32] = &[1, 2, 3];
    assert!(ys == zs);
}

#[test]
fn test_const_nonnull_new() {
    const {
        assert!(NonNull::new(core::ptr::null_mut::<()>()).is_none());

        let value = &mut 0u32;
        let mut ptr = NonNull::new(value).unwrap();
        unsafe { *ptr.as_mut() = 42 };

        let reference = unsafe { &*ptr.as_ref() };
        assert!(*reference == *value);
        assert!(*reference == 42);
    };
}

#[test]
#[cfg(unix)] // printf may not be available on other platforms
#[allow(deprecated)] // For SipHasher
pub fn test_variadic_fnptr() {
    use core::ffi;
    use core::hash::{Hash, SipHasher};
    extern "C" {
        // This needs to use the correct function signature even though it isn't called as some
        // codegen backends make it UB to declare a function with multiple conflicting signatures
        // (like LLVM) while others straight up return an error (like Cranelift).
        fn printf(_: *const ffi::c_char, ...) -> ffi::c_int;
    }
    let p: unsafe extern "C" fn(*const ffi::c_char, ...) -> ffi::c_int = printf;
    let q = p.clone();
    assert_eq!(p, q);
    assert!(!(p < q));
    let mut s = SipHasher::new();
    assert_eq!(p.hash(&mut s), q.hash(&mut s));
}

#[test]
fn write_unaligned_drop() {
    thread_local! {
        static DROPS: RefCell<Vec<u32>> = RefCell::new(Vec::new());
    }

    struct Dropper(u32);

    impl Drop for Dropper {
        fn drop(&mut self) {
            DROPS.with(|d| d.borrow_mut().push(self.0));
        }
    }

    {
        let c = Dropper(0);
        let mut t = Dropper(1);
        unsafe {
            write_unaligned(&mut t, c);
        }
    }
    DROPS.with(|d| assert_eq!(*d.borrow(), [0]));
}

#[test]
fn align_offset_zst() {
    // For pointers of stride = 0, the pointer is already aligned or it cannot be aligned at
    // all, because no amount of elements will align the pointer.
    let mut p = 1;
    while p < 1024 {
        assert_eq!(ptr::invalid::<()>(p).align_offset(p), 0);
        if p != 1 {
            assert_eq!(ptr::invalid::<()>(p + 1).align_offset(p), !0);
        }
        p = (p + 1).next_power_of_two();
    }
}

#[test]
fn align_offset_zst_const() {
    const {
        // For pointers of stride = 0, the pointer is already aligned or it cannot be aligned at
        // all, because no amount of elements will align the pointer.
        let mut p = 1;
        while p < 1024 {
            assert!(ptr::invalid::<()>(p).align_offset(p) == 0);
            if p != 1 {
                assert!(ptr::invalid::<()>(p + 1).align_offset(p) == !0);
            }
            p = (p + 1).next_power_of_two();
        }
    }
}

#[test]
fn align_offset_stride_one() {
    // For pointers of stride = 1, the pointer can always be aligned. The offset is equal to
    // number of bytes.
    let mut align = 1;
    while align < 1024 {
        for ptr in 1..2 * align {
            let expected = ptr % align;
            let offset = if expected == 0 { 0 } else { align - expected };
            assert_eq!(
                ptr::invalid::<u8>(ptr).align_offset(align),
                offset,
                "ptr = {}, align = {}, size = 1",
                ptr,
                align
            );
        }
        align = (align + 1).next_power_of_two();
    }
}

#[test]
fn align_offset_stride_one_const() {
    const {
        // For pointers of stride = 1, the pointer can always be aligned. The offset is equal to
        // number of bytes.
        let mut align = 1;
        while align < 1024 {
            let mut ptr = 1;
            while ptr < 2 * align {
                let expected = ptr % align;
                let offset = if expected == 0 { 0 } else { align - expected };
                assert!(ptr::invalid::<u8>(ptr).align_offset(align) == offset);
                ptr += 1;
            }
            align = (align + 1).next_power_of_two();
        }
    }
}

#[test]
fn align_offset_various_strides() {
    unsafe fn test_stride<T>(ptr: *const T, align: usize) -> bool {
        let numptr = ptr as usize;
        let mut expected = usize::MAX;
        // Naive but definitely correct way to find the *first* aligned element of stride::<T>.
        for el in 0..align {
            if (numptr + el * ::std::mem::size_of::<T>()) % align == 0 {
                expected = el;
                break;
            }
        }
        let got = ptr.align_offset(align);
        if got != expected {
            eprintln!(
                "aligning {:p} (with stride of {}) to {}, expected {}, got {}",
                ptr,
                ::std::mem::size_of::<T>(),
                align,
                expected,
                got
            );
            return true;
        }
        return false;
    }

    // For pointers of stride != 1, we verify the algorithm against the naivest possible
    // implementation
    let mut align = 1;
    let mut x = false;
    // Miri is too slow
    let limit = if cfg!(miri) { 32 } else { 1024 };
    while align < limit {
        for ptr in 1usize..4 * align {
            unsafe {
                #[repr(packed)]
                struct A3(u16, u8);
                x |= test_stride::<A3>(ptr::invalid::<A3>(ptr), align);

                struct A4(u32);
                x |= test_stride::<A4>(ptr::invalid::<A4>(ptr), align);

                #[repr(packed)]
                struct A5(u32, u8);
                x |= test_stride::<A5>(ptr::invalid::<A5>(ptr), align);

                #[repr(packed)]
                struct A6(u32, u16);
                x |= test_stride::<A6>(ptr::invalid::<A6>(ptr), align);

                #[repr(packed)]
                struct A7(u32, u16, u8);
                x |= test_stride::<A7>(ptr::invalid::<A7>(ptr), align);

                #[repr(packed)]
                struct A8(u32, u32);
                x |= test_stride::<A8>(ptr::invalid::<A8>(ptr), align);

                #[repr(packed)]
                struct A9(u32, u32, u8);
                x |= test_stride::<A9>(ptr::invalid::<A9>(ptr), align);

                #[repr(packed)]
                struct A10(u32, u32, u16);
                x |= test_stride::<A10>(ptr::invalid::<A10>(ptr), align);

                x |= test_stride::<u32>(ptr::invalid::<u32>(ptr), align);
                x |= test_stride::<u128>(ptr::invalid::<u128>(ptr), align);
            }
        }
        align = (align + 1).next_power_of_two();
    }
    assert!(!x);
}

#[test]
fn align_offset_various_strides_const() {
    const unsafe fn test_stride<T>(ptr: *const T, numptr: usize, align: usize) {
        let mut expected = usize::MAX;
        // Naive but definitely correct way to find the *first* aligned element of stride::<T>.
        let mut el = 0;
        while el < align {
            if (numptr + el * ::std::mem::size_of::<T>()) % align == 0 {
                expected = el;
                break;
            }
            el += 1;
        }
        let got = ptr.align_offset(align);
        assert!(got == expected);
    }

    const {
        // For pointers of stride != 1, we verify the algorithm against the naivest possible
        // implementation
        let mut align = 1;
        let limit = 32;
        while align < limit {
            let mut ptr = 1;
            while ptr < 4 * align {
                unsafe {
                    #[repr(packed)]
                    struct A3(u16, u8);
                    test_stride::<A3>(ptr::invalid::<A3>(ptr), ptr, align);

                    struct A4(u32);
                    test_stride::<A4>(ptr::invalid::<A4>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A5(u32, u8);
                    test_stride::<A5>(ptr::invalid::<A5>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A6(u32, u16);
                    test_stride::<A6>(ptr::invalid::<A6>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A7(u32, u16, u8);
                    test_stride::<A7>(ptr::invalid::<A7>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A8(u32, u32);
                    test_stride::<A8>(ptr::invalid::<A8>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A9(u32, u32, u8);
                    test_stride::<A9>(ptr::invalid::<A9>(ptr), ptr, align);

                    #[repr(packed)]
                    struct A10(u32, u32, u16);
                    test_stride::<A10>(ptr::invalid::<A10>(ptr), ptr, align);

                    test_stride::<u32>(ptr::invalid::<u32>(ptr), ptr, align);
                    test_stride::<u128>(ptr::invalid::<u128>(ptr), ptr, align);
                }
                ptr += 1;
            }
            align = (align + 1).next_power_of_two();
        }
    }
}

#[test]
fn align_offset_with_provenance_const() {
    const {
        // On some platforms (e.g. msp430-none-elf), the alignment of `i32` is less than 4.
        #[repr(align(4))]
        struct AlignedI32(i32);

        let data = AlignedI32(42);

        // `stride % align == 0` (usual case)

        let ptr: *const i32 = &data.0;
        assert!(ptr.align_offset(1) == 0);
        assert!(ptr.align_offset(2) == 0);
        assert!(ptr.align_offset(4) == 0);
        assert!(ptr.align_offset(8) == usize::MAX);
        assert!(ptr.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr.wrapping_byte_add(1).align_offset(2) == usize::MAX);
        assert!(ptr.wrapping_byte_add(2).align_offset(1) == 0);
        assert!(ptr.wrapping_byte_add(2).align_offset(2) == 0);
        assert!(ptr.wrapping_byte_add(2).align_offset(4) == usize::MAX);
        assert!(ptr.wrapping_byte_add(3).align_offset(1) == 0);
        assert!(ptr.wrapping_byte_add(3).align_offset(2) == usize::MAX);

        assert!(ptr.wrapping_add(42).align_offset(4) == 0);
        assert!(ptr.wrapping_add(42).align_offset(8) == usize::MAX);

        let ptr1: *const i8 = ptr.cast();
        assert!(ptr1.align_offset(1) == 0);
        assert!(ptr1.align_offset(2) == 0);
        assert!(ptr1.align_offset(4) == 0);
        assert!(ptr1.align_offset(8) == usize::MAX);
        assert!(ptr1.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr1.wrapping_byte_add(1).align_offset(2) == 1);
        assert!(ptr1.wrapping_byte_add(1).align_offset(4) == 3);
        assert!(ptr1.wrapping_byte_add(1).align_offset(8) == usize::MAX);
        assert!(ptr1.wrapping_byte_add(2).align_offset(1) == 0);
        assert!(ptr1.wrapping_byte_add(2).align_offset(2) == 0);
        assert!(ptr1.wrapping_byte_add(2).align_offset(4) == 2);
        assert!(ptr1.wrapping_byte_add(2).align_offset(8) == usize::MAX);
        assert!(ptr1.wrapping_byte_add(3).align_offset(1) == 0);
        assert!(ptr1.wrapping_byte_add(3).align_offset(2) == 1);
        assert!(ptr1.wrapping_byte_add(3).align_offset(4) == 1);
        assert!(ptr1.wrapping_byte_add(3).align_offset(8) == usize::MAX);

        let ptr2: *const i16 = ptr.cast();
        assert!(ptr2.align_offset(1) == 0);
        assert!(ptr2.align_offset(2) == 0);
        assert!(ptr2.align_offset(4) == 0);
        assert!(ptr2.align_offset(8) == usize::MAX);
        assert!(ptr2.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr2.wrapping_byte_add(1).align_offset(2) == usize::MAX);
        assert!(ptr2.wrapping_byte_add(2).align_offset(1) == 0);
        assert!(ptr2.wrapping_byte_add(2).align_offset(2) == 0);
        assert!(ptr2.wrapping_byte_add(2).align_offset(4) == 1);
        assert!(ptr2.wrapping_byte_add(2).align_offset(8) == usize::MAX);
        assert!(ptr2.wrapping_byte_add(3).align_offset(1) == 0);
        assert!(ptr2.wrapping_byte_add(3).align_offset(2) == usize::MAX);

        let ptr3: *const i64 = ptr.cast();
        assert!(ptr3.align_offset(1) == 0);
        assert!(ptr3.align_offset(2) == 0);
        assert!(ptr3.align_offset(4) == 0);
        assert!(ptr3.align_offset(8) == usize::MAX);
        assert!(ptr3.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr3.wrapping_byte_add(1).align_offset(2) == usize::MAX);

        // `stride % align != 0` (edge case)

        let ptr4: *const [u8; 3] = ptr.cast();
        assert!(ptr4.align_offset(1) == 0);
        assert!(ptr4.align_offset(2) == 0);
        assert!(ptr4.align_offset(4) == 0);
        assert!(ptr4.align_offset(8) == usize::MAX);
        assert!(ptr4.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr4.wrapping_byte_add(1).align_offset(2) == 1);
        assert!(ptr4.wrapping_byte_add(1).align_offset(4) == 1);
        assert!(ptr4.wrapping_byte_add(1).align_offset(8) == usize::MAX);
        assert!(ptr4.wrapping_byte_add(2).align_offset(1) == 0);
        assert!(ptr4.wrapping_byte_add(2).align_offset(2) == 0);
        assert!(ptr4.wrapping_byte_add(2).align_offset(4) == 2);
        assert!(ptr4.wrapping_byte_add(2).align_offset(8) == usize::MAX);
        assert!(ptr4.wrapping_byte_add(3).align_offset(1) == 0);
        assert!(ptr4.wrapping_byte_add(3).align_offset(2) == 1);
        assert!(ptr4.wrapping_byte_add(3).align_offset(4) == 3);
        assert!(ptr4.wrapping_byte_add(3).align_offset(8) == usize::MAX);

        let ptr5: *const [u8; 5] = ptr.cast();
        assert!(ptr5.align_offset(1) == 0);
        assert!(ptr5.align_offset(2) == 0);
        assert!(ptr5.align_offset(4) == 0);
        assert!(ptr5.align_offset(8) == usize::MAX);
        assert!(ptr5.wrapping_byte_add(1).align_offset(1) == 0);
        assert!(ptr5.wrapping_byte_add(1).align_offset(2) == 1);
        assert!(ptr5.wrapping_byte_add(1).align_offset(4) == 3);
        assert!(ptr5.wrapping_byte_add(1).align_offset(8) == usize::MAX);
        assert!(ptr5.wrapping_byte_add(2).align_offset(1) == 0);
        assert!(ptr5.wrapping_byte_add(2).align_offset(2) == 0);
        assert!(ptr5.wrapping_byte_add(2).align_offset(4) == 2);
        assert!(ptr5.wrapping_byte_add(2).align_offset(8) == usize::MAX);
        assert!(ptr5.wrapping_byte_add(3).align_offset(1) == 0);
        assert!(ptr5.wrapping_byte_add(3).align_offset(2) == 1);
        assert!(ptr5.wrapping_byte_add(3).align_offset(4) == 1);
        assert!(ptr5.wrapping_byte_add(3).align_offset(8) == usize::MAX);
    }
}

#[test]
fn align_offset_issue_103361() {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 1 << 47;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 1 << 30;
    #[cfg(target_pointer_width = "16")]
    const SIZE: usize = 1 << 13;
    struct HugeSize([u8; SIZE - 1]);
    let _ = ptr::invalid::<HugeSize>(SIZE).align_offset(SIZE);
}

#[test]
fn align_offset_issue_103361_const() {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 1 << 47;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 1 << 30;
    #[cfg(target_pointer_width = "16")]
    const SIZE: usize = 1 << 13;
    struct HugeSize([u8; SIZE - 1]);

    const {
        assert!(ptr::invalid::<HugeSize>(SIZE - 1).align_offset(SIZE) == SIZE - 1);
        assert!(ptr::invalid::<HugeSize>(SIZE).align_offset(SIZE) == 0);
        assert!(ptr::invalid::<HugeSize>(SIZE + 1).align_offset(SIZE) == 1);
    }
}

#[test]
fn is_aligned() {
    let data = 42;
    let ptr: *const i32 = &data;
    assert!(ptr.is_aligned());
    assert!(ptr.is_aligned_to(1));
    assert!(ptr.is_aligned_to(2));
    assert!(ptr.is_aligned_to(4));
    assert!(ptr.wrapping_byte_add(2).is_aligned_to(1));
    assert!(ptr.wrapping_byte_add(2).is_aligned_to(2));
    assert!(!ptr.wrapping_byte_add(2).is_aligned_to(4));

    // At runtime either `ptr` or `ptr+1` is aligned to 8.
    assert_ne!(ptr.is_aligned_to(8), ptr.wrapping_add(1).is_aligned_to(8));
}

#[test]
fn is_aligned_const() {
    const {
        let data = 42;
        let ptr: *const i32 = &data;
        assert!(ptr.is_aligned());
        assert!(ptr.is_aligned_to(1));
        assert!(ptr.is_aligned_to(2));
        assert!(ptr.is_aligned_to(4));
        assert!(ptr.wrapping_byte_add(2).is_aligned_to(1));
        assert!(ptr.wrapping_byte_add(2).is_aligned_to(2));
        assert!(!ptr.wrapping_byte_add(2).is_aligned_to(4));

        // At comptime neither `ptr` nor `ptr+1` is aligned to 8.
        assert!(!ptr.is_aligned_to(8));
        assert!(!ptr.wrapping_add(1).is_aligned_to(8));
    }
}

#[test]
fn offset_from() {
    let mut a = [0; 5];
    let ptr1: *mut i32 = &mut a[1];
    let ptr2: *mut i32 = &mut a[3];
    unsafe {
        assert_eq!(ptr2.offset_from(ptr1), 2);
        assert_eq!(ptr1.offset_from(ptr2), -2);
        assert_eq!(ptr1.offset(2), ptr2);
        assert_eq!(ptr2.offset(-2), ptr1);
    }
}

#[test]
fn ptr_metadata() {
    struct Unit;
    struct Pair<A, B: ?Sized>(A, B);
    extern "C" {
        type Extern;
    }
    let () = metadata(&());
    let () = metadata(&Unit);
    let () = metadata(&4_u32);
    let () = metadata(&String::new());
    let () = metadata(&Some(4_u32));
    let () = metadata(&ptr_metadata);
    let () = metadata(&|| {});
    let () = metadata(&[4, 7]);
    let () = metadata(&(4, String::new()));
    let () = metadata(&Pair(4, String::new()));
    let () = metadata(ptr::null::<()>() as *const Extern);
    let () = metadata(ptr::null::<()>() as *const <&u32 as std::ops::Deref>::Target);

    assert_eq!(metadata("foo"), 3_usize);
    assert_eq!(metadata(&[4, 7][..]), 2_usize);

    let dst_tuple: &(bool, [u8]) = &(true, [0x66, 0x6F, 0x6F]);
    let dst_struct: &Pair<bool, [u8]> = &Pair(true, [0x66, 0x6F, 0x6F]);
    assert_eq!(metadata(dst_tuple), 3_usize);
    assert_eq!(metadata(dst_struct), 3_usize);
    unsafe {
        let dst_tuple: &(bool, str) = std::mem::transmute(dst_tuple);
        let dst_struct: &Pair<bool, str> = std::mem::transmute(dst_struct);
        assert_eq!(&dst_tuple.1, "foo");
        assert_eq!(&dst_struct.1, "foo");
        assert_eq!(metadata(dst_tuple), 3_usize);
        assert_eq!(metadata(dst_struct), 3_usize);
    }

    let vtable_1: DynMetadata<dyn Debug> = metadata(&4_u16 as &dyn Debug);
    let vtable_2: DynMetadata<dyn Display> = metadata(&4_u16 as &dyn Display);
    let vtable_3: DynMetadata<dyn Display> = metadata(&4_u32 as &dyn Display);
    let vtable_4: DynMetadata<dyn Display> = metadata(&(true, 7_u32) as &(bool, dyn Display));
    let vtable_5: DynMetadata<dyn Display> =
        metadata(&Pair(true, 7_u32) as &Pair<bool, dyn Display>);
    unsafe {
        let address_1: *const () = std::mem::transmute(vtable_1);
        let address_2: *const () = std::mem::transmute(vtable_2);
        let address_3: *const () = std::mem::transmute(vtable_3);
        let address_4: *const () = std::mem::transmute(vtable_4);
        let address_5: *const () = std::mem::transmute(vtable_5);
        // Different trait => different vtable pointer
        assert_ne!(address_1, address_2);
        // Different erased type => different vtable pointer
        assert_ne!(address_2, address_3);
        // Same erased type and same trait => same vtable pointer
        assert_eq!(address_3, address_4);
        assert_eq!(address_3, address_5);
    }
}

#[test]
fn ptr_metadata_bounds() {
    fn metadata_eq_method_address<T: ?Sized>() -> usize {
        // The `Metadata` associated type has an `Ord` bound, so this is valid:
        <<T as Pointee>::Metadata as PartialEq>::eq as usize
    }
    // "Synthetic" trait impls generated by the compiler like those of `Pointee`
    // are not checked for bounds of associated type.
    // So with a buggy core we could have both:
    // * `<dyn Display as Pointee>::Metadata == DynMetadata`
    // * `DynMetadata: !PartialEq`
    // … and cause an ICE here:
    metadata_eq_method_address::<dyn Display>();

    // For this reason, let’s check here that bounds are satisfied:

    let _ = static_assert_expected_bounds_for_metadata::<()>;
    let _ = static_assert_expected_bounds_for_metadata::<usize>;
    let _ = static_assert_expected_bounds_for_metadata::<DynMetadata<dyn Display>>;
    fn _static_assert_associated_type<T: ?Sized>() {
        let _ = static_assert_expected_bounds_for_metadata::<<T as Pointee>::Metadata>;
    }

    fn static_assert_expected_bounds_for_metadata<Meta>()
    where
        // Keep this in sync with the associated type in `library/core/src/ptr/metadata.rs`
        Meta: Copy + Send + Sync + Ord + std::hash::Hash + Unpin,
    {
    }
}

#[test]
fn dyn_metadata() {
    #[derive(Debug)]
    #[repr(align(32))]
    struct Something([u8; 47]);

    let value = Something([0; 47]);
    let trait_object: &dyn Debug = &value;
    let meta = metadata(trait_object);

    assert_eq!(meta.size_of(), 64);
    assert_eq!(meta.size_of(), std::mem::size_of::<Something>());
    assert_eq!(meta.align_of(), 32);
    assert_eq!(meta.align_of(), std::mem::align_of::<Something>());
    assert_eq!(meta.layout(), std::alloc::Layout::new::<Something>());

    assert!(format!("{meta:?}").starts_with("DynMetadata(0x"));
}

#[test]
fn from_raw_parts() {
    let mut value = 5_u32;
    let address = &mut value as *mut _ as *mut ();
    let trait_object: &dyn Display = &mut value;
    let vtable = metadata(trait_object);
    let trait_object = NonNull::from(trait_object);

    assert_eq!(ptr::from_raw_parts(address, vtable), trait_object.as_ptr());
    assert_eq!(ptr::from_raw_parts_mut(address, vtable), trait_object.as_ptr());
    assert_eq!(NonNull::from_raw_parts(NonNull::new(address).unwrap(), vtable), trait_object);

    let mut array = [5_u32, 5, 5, 5, 5];
    let address = &mut array as *mut _ as *mut ();
    let array_ptr = NonNull::from(&mut array);
    let slice_ptr = NonNull::from(&mut array[..]);

    assert_eq!(ptr::from_raw_parts(address, ()), array_ptr.as_ptr());
    assert_eq!(ptr::from_raw_parts_mut(address, ()), array_ptr.as_ptr());
    assert_eq!(NonNull::from_raw_parts(NonNull::new(address).unwrap(), ()), array_ptr);

    assert_eq!(ptr::from_raw_parts(address, 5), slice_ptr.as_ptr());
    assert_eq!(ptr::from_raw_parts_mut(address, 5), slice_ptr.as_ptr());
    assert_eq!(NonNull::from_raw_parts(NonNull::new(address).unwrap(), 5), slice_ptr);
}

#[test]
fn thin_box() {
    let foo = ThinBox::<dyn Display>::new(4);
    assert_eq!(foo.to_string(), "4");
    drop(foo);
    let bar = ThinBox::<dyn Display>::new(7);
    assert_eq!(bar.to_string(), "7");

    // A slightly more interesting library that could be built on top of metadata APIs.
    //
    // * It could be generalized to any `T: ?Sized` (not just trait object)
    //   if `{size,align}_of_for_meta<T: ?Sized>(T::Metadata)` are added.
    // * Constructing a `ThinBox` without consuming and deallocating a `Box`
    //   requires either the unstable `Unsize` marker trait,
    //   or the unstable `unsized_locals` language feature,
    //   or taking `&dyn T` and restricting to `T: Copy`.

    use std::alloc::*;
    use std::marker::PhantomData;

    struct ThinBox<T>
    where
        T: ?Sized + Pointee<Metadata = DynMetadata<T>>,
    {
        ptr: NonNull<DynMetadata<T>>,
        phantom: PhantomData<T>,
    }

    impl<T> ThinBox<T>
    where
        T: ?Sized + Pointee<Metadata = DynMetadata<T>>,
    {
        pub fn new<Value: std::marker::Unsize<T>>(value: Value) -> Self {
            let unsized_: &T = &value;
            let meta = metadata(unsized_);
            let meta_layout = Layout::for_value(&meta);
            let value_layout = Layout::for_value(&value);
            let (layout, offset) = meta_layout.extend(value_layout).unwrap();
            // `DynMetadata` is pointer-sized:
            assert!(layout.size() > 0);
            // If `ThinBox<T>` is generalized to any `T: ?Sized`,
            // handle ZSTs with a dangling pointer without going through `alloc()`,
            // like `Box<T>` does.
            unsafe {
                let ptr = NonNull::new(alloc(layout))
                    .unwrap_or_else(|| handle_alloc_error(layout))
                    .cast::<DynMetadata<T>>();
                ptr.as_ptr().write(meta);
                ptr.as_ptr().byte_add(offset).cast::<Value>().write(value);
                Self { ptr, phantom: PhantomData }
            }
        }

        fn meta(&self) -> DynMetadata<T> {
            unsafe { *self.ptr.as_ref() }
        }

        fn layout(&self) -> (Layout, usize) {
            let meta = self.meta();
            Layout::for_value(&meta).extend(meta.layout()).unwrap()
        }

        fn value_ptr(&self) -> *const T {
            let (_, offset) = self.layout();
            let data_ptr = unsafe { self.ptr.cast::<u8>().as_ptr().add(offset) };
            ptr::from_raw_parts(data_ptr.cast(), self.meta())
        }

        fn value_mut_ptr(&mut self) -> *mut T {
            let (_, offset) = self.layout();
            // FIXME: can this line be shared with the same in `value_ptr()`
            // without upsetting Stacked Borrows?
            let data_ptr = unsafe { self.ptr.cast::<u8>().as_ptr().add(offset) };
            from_raw_parts_mut(data_ptr.cast(), self.meta())
        }
    }

    impl<T> std::ops::Deref for ThinBox<T>
    where
        T: ?Sized + Pointee<Metadata = DynMetadata<T>>,
    {
        type Target = T;

        fn deref(&self) -> &T {
            unsafe { &*self.value_ptr() }
        }
    }

    impl<T> std::ops::DerefMut for ThinBox<T>
    where
        T: ?Sized + Pointee<Metadata = DynMetadata<T>>,
    {
        fn deref_mut(&mut self) -> &mut T {
            unsafe { &mut *self.value_mut_ptr() }
        }
    }

    impl<T> std::ops::Drop for ThinBox<T>
    where
        T: ?Sized + Pointee<Metadata = DynMetadata<T>>,
    {
        fn drop(&mut self) {
            let (layout, _) = self.layout();
            unsafe {
                drop_in_place::<T>(&mut **self);
                dealloc(self.ptr.cast().as_ptr(), layout);
            }
        }
    }
}

#[test]
fn nonnull_tagged_pointer_with_provenance() {
    let raw_pointer = Box::into_raw(Box::new(10));

    let mut p = TaggedPointer::new(raw_pointer).unwrap();
    assert_eq!(p.tag(), 0);

    p.set_tag(1);
    assert_eq!(p.tag(), 1);
    assert_eq!(unsafe { *p.pointer().as_ptr() }, 10);

    p.set_tag(3);
    assert_eq!(p.tag(), 3);
    assert_eq!(unsafe { *p.pointer().as_ptr() }, 10);

    unsafe { drop(Box::from_raw(p.pointer().as_ptr())) };

    /// A non-null pointer type which carries several bits of metadata and maintains provenance.
    #[repr(transparent)]
    pub struct TaggedPointer<T>(NonNull<T>);

    impl<T> Clone for TaggedPointer<T> {
        fn clone(&self) -> Self {
            Self(self.0)
        }
    }

    impl<T> Copy for TaggedPointer<T> {}

    impl<T> TaggedPointer<T> {
        /// The ABI-required minimum alignment of the `P` type.
        pub const ALIGNMENT: usize = core::mem::align_of::<T>();
        /// A mask for data-carrying bits of the address.
        pub const DATA_MASK: usize = !Self::ADDRESS_MASK;
        /// Number of available bits of storage in the address.
        pub const NUM_BITS: u32 = Self::ALIGNMENT.trailing_zeros();
        /// A mask for the non-data-carrying bits of the address.
        pub const ADDRESS_MASK: usize = usize::MAX << Self::NUM_BITS;

        /// Create a new tagged pointer from a possibly null pointer.
        pub fn new(pointer: *mut T) -> Option<TaggedPointer<T>> {
            Some(TaggedPointer(NonNull::new(pointer)?))
        }

        /// Consume this tagged pointer and produce a raw mutable pointer to the
        /// memory location.
        pub fn pointer(self) -> NonNull<T> {
            // SAFETY: The `addr` guaranteed to have bits set in the Self::ADDRESS_MASK, so the result will be non-null.
            self.0.map_addr(|addr| unsafe {
                NonZeroUsize::new_unchecked(addr.get() & Self::ADDRESS_MASK)
            })
        }

        /// Consume this tagged pointer and produce the data it carries.
        pub fn tag(&self) -> usize {
            self.0.addr().get() & Self::DATA_MASK
        }

        /// Update the data this tagged pointer carries to a new value.
        pub fn set_tag(&mut self, data: usize) {
            assert_eq!(
                data & Self::ADDRESS_MASK,
                0,
                "cannot set more data beyond the lowest NUM_BITS"
            );
            let data = data & Self::DATA_MASK;

            // SAFETY: This value will always be non-zero because the upper bits (from
            // ADDRESS_MASK) will always be non-zero. This a property of the type and its
            // construction.
            self.0 = self.0.map_addr(|addr| unsafe {
                NonZeroUsize::new_unchecked((addr.get() & Self::ADDRESS_MASK) | data)
            })
        }
    }
}

#[test]
fn swap_copy_untyped() {
    // We call `{swap,copy}{,_nonoverlapping}` at `bool` type on data that is not a valid bool.
    // These should all do untyped copies, so this should work fine.
    let mut x = 5u8;
    let mut y = 6u8;

    let ptr1 = addr_of_mut!(x).cast::<bool>();
    let ptr2 = addr_of_mut!(y).cast::<bool>();

    unsafe {
        ptr::swap(ptr1, ptr2);
        ptr::swap_nonoverlapping(ptr1, ptr2, 1);
    }
    assert_eq!(x, 5);
    assert_eq!(y, 6);

    unsafe {
        ptr::copy(ptr1, ptr2, 1);
        ptr::copy_nonoverlapping(ptr1, ptr2, 1);
    }
    assert_eq!(x, 5);
    assert_eq!(y, 5);
}

#[test]
fn test_const_copy() {
    const {
        let ptr1 = &1;
        let mut ptr2 = &666;

        // Copy ptr1 to ptr2, bytewise.
        unsafe {
            ptr::copy(
                &ptr1 as *const _ as *const MaybeUninit<u8>,
                &mut ptr2 as *mut _ as *mut MaybeUninit<u8>,
                mem::size_of::<&i32>(),
            );
        }

        // Make sure they still work.
        assert!(*ptr1 == 1);
        assert!(*ptr2 == 1);
    };

    const {
        let ptr1 = &1;
        let mut ptr2 = &666;

        // Copy ptr1 to ptr2, bytewise.
        unsafe {
            ptr::copy_nonoverlapping(
                &ptr1 as *const _ as *const MaybeUninit<u8>,
                &mut ptr2 as *mut _ as *mut MaybeUninit<u8>,
                mem::size_of::<&i32>(),
            );
        }

        // Make sure they still work.
        assert!(*ptr1 == 1);
        assert!(*ptr2 == 1);
    };
}
