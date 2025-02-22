use core::cell::RefCell;
use core::marker::Freeze;
use core::mem::MaybeUninit;
use core::num::NonZero;
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
        assert!(v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16);
        copy(v0.as_ptr().offset(2), v1.as_mut_ptr(), 1);
        assert!(v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 0u16);
        copy(v0.as_ptr(), v1.as_mut_ptr().offset(2), 1);
        assert!(v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 32000u16);
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

    unsafe extern "C" {
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
#[allow(unpredictable_function_pointer_comparisons)]
pub fn test_variadic_fnptr() {
    use core::ffi;
    use core::hash::{Hash, SipHasher};
    unsafe extern "C" {
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
        assert_eq!(ptr::without_provenance::<()>(p).align_offset(p), 0);
        if p != 1 {
            assert_eq!(ptr::without_provenance::<()>(p + 1).align_offset(p), !0);
        }
        p = (p + 1).next_power_of_two();
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
                ptr::without_provenance::<u8>(ptr).align_offset(align),
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
fn align_offset_various_strides() {
    unsafe fn test_stride<T>(ptr: *const T, align: usize) -> bool {
        let numptr = ptr as usize;
        let mut expected = usize::MAX;
        // Naive but definitely correct way to find the *first* aligned element of stride::<T>.
        for el in 0..align {
            if (numptr + el * size_of::<T>()) % align == 0 {
                expected = el;
                break;
            }
        }
        let got = ptr.align_offset(align);
        if got != expected {
            eprintln!(
                "aligning {:p} (with stride of {}) to {}, expected {}, got {}",
                ptr,
                size_of::<T>(),
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
                struct A3(#[allow(dead_code)] u16, #[allow(dead_code)] u8);
                x |= test_stride::<A3>(ptr::without_provenance::<A3>(ptr), align);

                struct A4(#[allow(dead_code)] u32);
                x |= test_stride::<A4>(ptr::without_provenance::<A4>(ptr), align);

                #[repr(packed)]
                struct A5(#[allow(dead_code)] u32, #[allow(dead_code)] u8);
                x |= test_stride::<A5>(ptr::without_provenance::<A5>(ptr), align);

                #[repr(packed)]
                struct A6(#[allow(dead_code)] u32, #[allow(dead_code)] u16);
                x |= test_stride::<A6>(ptr::without_provenance::<A6>(ptr), align);

                #[repr(packed)]
                struct A7(#[allow(dead_code)] u32, #[allow(dead_code)] u16, #[allow(dead_code)] u8);
                x |= test_stride::<A7>(ptr::without_provenance::<A7>(ptr), align);

                #[repr(packed)]
                struct A8(#[allow(dead_code)] u32, #[allow(dead_code)] u32);
                x |= test_stride::<A8>(ptr::without_provenance::<A8>(ptr), align);

                #[repr(packed)]
                struct A9(#[allow(dead_code)] u32, #[allow(dead_code)] u32, #[allow(dead_code)] u8);
                x |= test_stride::<A9>(ptr::without_provenance::<A9>(ptr), align);

                #[repr(packed)]
                struct A10(
                    #[allow(dead_code)] u32,
                    #[allow(dead_code)] u32,
                    #[allow(dead_code)] u16,
                );
                x |= test_stride::<A10>(ptr::without_provenance::<A10>(ptr), align);

                x |= test_stride::<u32>(ptr::without_provenance::<u32>(ptr), align);
                x |= test_stride::<u128>(ptr::without_provenance::<u128>(ptr), align);
            }
        }
        align = (align + 1).next_power_of_two();
    }
    assert!(!x);
}

#[test]
fn align_offset_issue_103361() {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 1 << 47;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 1 << 30;
    #[cfg(target_pointer_width = "16")]
    const SIZE: usize = 1 << 13;
    struct HugeSize(#[allow(dead_code)] [u8; SIZE - 1]);
    let _ = ptr::without_provenance::<HugeSize>(SIZE).align_offset(SIZE);
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
    unsafe extern "C" {
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

    let dst_struct: &Pair<bool, [u8]> = &Pair(true, [0x66, 0x6F, 0x6F]);
    assert_eq!(metadata(dst_struct), 3_usize);
    unsafe {
        let dst_struct: &Pair<bool, str> = std::mem::transmute(dst_struct);
        assert_eq!(&dst_struct.1, "foo");
        assert_eq!(metadata(dst_struct), 3_usize);
    }

    let vtable_1: DynMetadata<dyn Debug> = metadata(&4_u16 as &dyn Debug);
    let vtable_2: DynMetadata<dyn Display> = metadata(&4_u16 as &dyn Display);
    let vtable_3: DynMetadata<dyn Display> = metadata(&4_u32 as &dyn Display);
    let vtable_4: DynMetadata<dyn Display> =
        metadata(&Pair(true, 7_u32) as &Pair<bool, dyn Display>);
    unsafe {
        let address_1: *const () = std::mem::transmute(vtable_1);
        let address_2: *const () = std::mem::transmute(vtable_2);
        let address_3: *const () = std::mem::transmute(vtable_3);
        let address_4: *const () = std::mem::transmute(vtable_4);
        // Different trait => different vtable pointer
        assert_ne!(address_1, address_2);
        // Different erased type => different vtable pointer
        assert_ne!(address_2, address_3);
        // Same erased type and same trait => same vtable pointer.
        // This is *not guaranteed*, so we skip it in Miri.
        if !cfg!(miri) {
            assert_eq!(address_3, address_4);
        }
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
        Meta: Debug + Copy + Send + Sync + Ord + std::hash::Hash + Unpin + Freeze,
    {
    }
}

#[test]
fn pointee_metadata_debug() {
    assert_eq!("()", format!("{:?}", metadata::<u32>(&17)));
    assert_eq!("2", format!("{:?}", metadata::<[u32]>(&[19, 23])));
    let for_dyn = format!("{:?}", metadata::<dyn Debug>(&29));
    assert!(for_dyn.starts_with("DynMetadata(0x"), "{:?}", for_dyn);
}

#[test]
fn dyn_metadata() {
    #[derive(Debug)]
    #[repr(align(32))]
    struct Something(#[allow(dead_code)] [u8; 47]);

    let value = Something([0; 47]);
    let trait_object: &dyn Debug = &value;
    let meta = metadata(trait_object);

    assert_eq!(meta.size_of(), 64);
    assert_eq!(meta.size_of(), size_of::<Something>());
    assert_eq!(meta.align_of(), 32);
    assert_eq!(meta.align_of(), align_of::<Something>());
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
            ptr::from_raw_parts(data_ptr, self.meta())
        }

        fn value_mut_ptr(&mut self) -> *mut T {
            let (_, offset) = self.layout();
            // FIXME: can this line be shared with the same in `value_ptr()`
            // without upsetting Stacked Borrows?
            let data_ptr = unsafe { self.ptr.cast::<u8>().as_ptr().add(offset) };
            from_raw_parts_mut(data_ptr, self.meta())
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
        pub const ALIGNMENT: usize = align_of::<T>();
        /// A mask for data-carrying bits of the address.
        pub const DATA_MASK: usize = !Self::ADDRESS_MASK;
        /// Number of available bits of storage in the address.
        pub const NUM_BITS: u32 = Self::ALIGNMENT.trailing_zeros();
        /// A mask for the non-data-carrying bits of the address.
        pub const ADDRESS_MASK: usize = usize::MAX << Self::NUM_BITS;

        /// Creates a new tagged pointer from a possibly null pointer.
        pub fn new(pointer: *mut T) -> Option<TaggedPointer<T>> {
            Some(TaggedPointer(NonNull::new(pointer)?))
        }

        /// Consume this tagged pointer and produce a raw mutable pointer to the
        /// memory location.
        pub fn pointer(self) -> NonNull<T> {
            // SAFETY: The `addr` guaranteed to have bits set in the Self::ADDRESS_MASK, so the result will be non-null.
            self.0
                .map_addr(|addr| unsafe { NonZero::new_unchecked(addr.get() & Self::ADDRESS_MASK) })
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
                NonZero::new_unchecked((addr.get() & Self::ADDRESS_MASK) | data)
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
fn test_const_copy_ptr() {
    // `copy` and `copy_nonoverlapping` are thin layers on top of intrinsics. Ensure they correctly
    // deal with pointers even when the pointers cross the boundary from one "element" being copied
    // to another.
    const {
        let ptr1 = &1;
        let mut ptr2 = &666;

        // Copy ptr1 to ptr2, bytewise.
        unsafe {
            ptr::copy(
                &ptr1 as *const _ as *const MaybeUninit<u8>,
                &mut ptr2 as *mut _ as *mut MaybeUninit<u8>,
                size_of::<&i32>(),
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
                size_of::<&i32>(),
            );
        }

        // Make sure they still work.
        assert!(*ptr1 == 1);
        assert!(*ptr2 == 1);
    };
}

#[test]
fn test_const_swap_ptr() {
    // The `swap` functions are implemented in the library, they are not primitives.
    // Only `swap_nonoverlapping` takes a count; pointers that cross multiple elements
    // are *not* supported.
    // We put the pointer at an odd offset in the type and copy them as an array of bytes,
    // which should catch most of the ways that the library implementation can get it wrong.

    #[cfg(target_pointer_width = "32")]
    type HalfPtr = i16;
    #[cfg(target_pointer_width = "64")]
    type HalfPtr = i32;

    #[repr(C, packed)]
    #[allow(unused)]
    struct S {
        f1: HalfPtr,
        // Crucially this field is at an offset that is not a multiple of the pointer size.
        ptr: &'static i32,
        // Make sure the entire type does not have a power-of-2 size:
        // make it 3 pointers in size. This used to hit a bug in `swap_nonoverlapping`.
        f2: [HalfPtr; 3],
    }

    // Ensure the entire thing is usize-aligned, so in principle this
    // looks like it could be eligible for a `usize` copying loop.
    #[cfg_attr(target_pointer_width = "32", repr(align(4)))]
    #[cfg_attr(target_pointer_width = "64", repr(align(8)))]
    struct A(S);

    const {
        let mut s1 = A(S { ptr: &1, f1: 0, f2: [0; 3] });
        let mut s2 = A(S { ptr: &666, f1: 0, f2: [0; 3] });

        // Swap ptr1 and ptr2, as an array.
        type T = [u8; size_of::<A>()];
        unsafe {
            ptr::swap(ptr::from_mut(&mut s1).cast::<T>(), ptr::from_mut(&mut s2).cast::<T>());
        }

        // Make sure they still work.
        assert!(*s1.0.ptr == 666);
        assert!(*s2.0.ptr == 1);

        // Swap them back, again as an array.
        unsafe {
            ptr::swap_nonoverlapping(
                ptr::from_mut(&mut s1).cast::<T>(),
                ptr::from_mut(&mut s2).cast::<T>(),
                1,
            );
        }

        // Make sure they still work.
        assert!(*s1.0.ptr == 1);
        assert!(*s2.0.ptr == 666);
    };
}

#[test]
fn test_null_array_as_slice() {
    let arr: *mut [u8; 4] = null_mut();
    let ptr: *mut [u8] = arr.as_mut_slice();
    assert!(ptr.is_null());
    assert_eq!(ptr.len(), 4);

    let arr: *const [u8; 4] = null();
    let ptr: *const [u8] = arr.as_slice();
    assert!(ptr.is_null());
    assert_eq!(ptr.len(), 4);
}

#[test]
fn test_ptr_from_raw_parts_in_const() {
    const EMPTY_SLICE_PTR: *const [i32] =
        std::ptr::slice_from_raw_parts(std::ptr::without_provenance(123), 456);
    assert_eq!(EMPTY_SLICE_PTR.addr(), 123);
    assert_eq!(EMPTY_SLICE_PTR.len(), 456);
}

#[test]
fn test_ptr_metadata_in_const() {
    use std::fmt::Debug;

    const ARRAY_META: () = std::ptr::metadata::<[u16; 3]>(&[1, 2, 3]);
    const SLICE_META: usize = std::ptr::metadata::<[u16]>(&[1, 2, 3]);
    const DYN_META: DynMetadata<dyn Debug> = std::ptr::metadata::<dyn Debug>(&[0_u8; 42]);
    assert_eq!(ARRAY_META, ());
    assert_eq!(SLICE_META, 3);
    assert_eq!(DYN_META.size_of(), 42);
}

// See <https://github.com/rust-lang/rust/issues/134713>
const fn ptr_swap_nonoverlapping_is_untyped_inner() {
    #[repr(C)]
    struct HasPadding(usize, u8);

    let buf1: [usize; 2] = [1000, 2000];
    let buf2: [usize; 2] = [3000, 4000];

    // HasPadding and [usize; 2] have the same size and alignment,
    // so swap_nonoverlapping should treat them the same
    assert!(size_of::<HasPadding>() == size_of::<[usize; 2]>());
    assert!(align_of::<HasPadding>() == align_of::<[usize; 2]>());

    let mut b1 = buf1;
    let mut b2 = buf2;
    // Safety: b1 and b2 are distinct local variables,
    // with the same size and alignment as HasPadding.
    unsafe {
        std::ptr::swap_nonoverlapping(
            b1.as_mut_ptr().cast::<HasPadding>(),
            b2.as_mut_ptr().cast::<HasPadding>(),
            1,
        );
    }
    assert!(b1[0] == buf2[0]);
    assert!(b1[1] == buf2[1]);
    assert!(b2[0] == buf1[0]);
    assert!(b2[1] == buf1[1]);
}

#[test]
fn test_ptr_swap_nonoverlapping_is_untyped() {
    ptr_swap_nonoverlapping_is_untyped_inner();
    const { ptr_swap_nonoverlapping_is_untyped_inner() };
}
