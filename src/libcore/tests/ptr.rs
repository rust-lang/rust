use core::ptr::*;
use core::cell::RefCell;

#[test]
fn test() {
    unsafe {
        struct Pair {
            fst: isize,
            snd: isize
        };
        let mut p = Pair {fst: 10, snd: 20};
        let pptr: *mut Pair = &mut p;
        let iptr: *mut isize = pptr as *mut isize;
        assert_eq!(*iptr, 10);
        *iptr = 30;
        assert_eq!(*iptr, 30);
        assert_eq!(p.fst, 30);

        *pptr = Pair {fst: 50, snd: 60};
        assert_eq!(*iptr, 50);
        assert_eq!(p.fst, 50);
        assert_eq!(p.snd, 60);

        let v0 = vec![32000u16, 32001u16, 32002u16];
        let mut v1 = vec![0u16, 0u16, 0u16];

        copy(v0.as_ptr().offset(1), v1.as_mut_ptr().offset(1), 1);
        assert!((v1[0] == 0u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy(v0.as_ptr().offset(2), v1.as_mut_ptr(), 1);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy(v0.as_ptr(), v1.as_mut_ptr().offset(2), 1);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 32000u16));
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
#[cfg(not(miri))] // This test is UB according to Stacked Borrows
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
        assert_eq!(ms.as_mut(), Some(s));

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
        let xs = vec![0,1,2,3,4,5,6,7,8,9];
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

        assert_eq!(xs_mut, [0,2,4,6,8,10,12,14,16,18]);
    }
}

#[test]
fn test_set_memory() {
    let mut xs = [0u8; 20];
    let ptr = xs.as_mut_ptr();
    unsafe { write_bytes(ptr, 5u8, xs.len()); }
    assert!(xs == [5u8; 20]);
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
#[allow(warnings)]
// Have a symbol for the test below. It doesn’t need to be an actual variadic function, match the
// ABI, or even point to an actual executable code, because the function itself is never invoked.
#[no_mangle]
pub fn test_variadic_fnptr() {
    use core::hash::{Hash, SipHasher};
    extern {
        fn test_variadic_fnptr(_: u64, ...) -> f64;
    }
    let p: unsafe extern fn(u64, ...) -> f64 = test_variadic_fnptr;
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
        unsafe { write_unaligned(&mut t, c); }
    }
    DROPS.with(|d| assert_eq!(*d.borrow(), [0]));
}

#[test]
#[cfg(not(miri))] // Miri does not compute a maximal `mid` for `align_offset`
fn align_offset_zst() {
    // For pointers of stride = 0, the pointer is already aligned or it cannot be aligned at
    // all, because no amount of elements will align the pointer.
    let mut p = 1;
    while p < 1024 {
        assert_eq!((p as *const ()).align_offset(p), 0);
        if p != 1 {
            assert_eq!(((p + 1) as *const ()).align_offset(p), !0);
        }
        p = (p + 1).next_power_of_two();
    }
}

#[test]
#[cfg(not(miri))] // Miri does not compute a maximal `mid` for `align_offset`
fn align_offset_stride1() {
    // For pointers of stride = 1, the pointer can always be aligned. The offset is equal to
    // number of bytes.
    let mut align = 1;
    while align < 1024 {
        for ptr in 1..2*align {
            let expected = ptr % align;
            let offset = if expected == 0 { 0 } else { align - expected };
            assert_eq!((ptr as *const u8).align_offset(align), offset,
            "ptr = {}, align = {}, size = 1", ptr, align);
        }
        align = (align + 1).next_power_of_two();
    }
}

#[test]
#[cfg(not(miri))] // Miri is too slow
fn align_offset_weird_strides() {
    #[repr(packed)]
    struct A3(u16, u8);
    struct A4(u32);
    #[repr(packed)]
    struct A5(u32, u8);
    #[repr(packed)]
    struct A6(u32, u16);
    #[repr(packed)]
    struct A7(u32, u16, u8);
    #[repr(packed)]
    struct A8(u32, u32);
    #[repr(packed)]
    struct A9(u32, u32, u8);
    #[repr(packed)]
    struct A10(u32, u32, u16);

    unsafe fn test_weird_stride<T>(ptr: *const T, align: usize) -> bool {
        let numptr = ptr as usize;
        let mut expected = usize::max_value();
        // Naive but definitely correct way to find the *first* aligned element of stride::<T>.
        for el in 0..align {
            if (numptr + el * ::std::mem::size_of::<T>()) % align == 0 {
                expected = el;
                break;
            }
        }
        let got = ptr.align_offset(align);
        if got != expected {
            eprintln!("aligning {:p} (with stride of {}) to {}, expected {}, got {}", ptr,
                      ::std::mem::size_of::<T>(), align, expected, got);
            return true;
        }
        return false;
    }

    // For pointers of stride != 1, we verify the algorithm against the naivest possible
    // implementation
    let mut align = 1;
    let mut x = false;
    while align < 1024 {
        for ptr in 1usize..4*align {
            unsafe {
                x |= test_weird_stride::<A3>(ptr as *const A3, align);
                x |= test_weird_stride::<A4>(ptr as *const A4, align);
                x |= test_weird_stride::<A5>(ptr as *const A5, align);
                x |= test_weird_stride::<A6>(ptr as *const A6, align);
                x |= test_weird_stride::<A7>(ptr as *const A7, align);
                x |= test_weird_stride::<A8>(ptr as *const A8, align);
                x |= test_weird_stride::<A9>(ptr as *const A9, align);
                x |= test_weird_stride::<A10>(ptr as *const A10, align);
            }
        }
        align = (align + 1).next_power_of_two();
    }
    assert!(!x);
}
