//@compile-flags: -Zmiri-permissive-provenance
use std::{mem, ptr};

fn main() {
    smoke();
    test_offset_from();
    test_vec_into_iter();
    ptr_arith_offset();
    ptr_arith_offset_overflow();
    ptr_offset();
}

fn smoke() {
    // Smoke-test various offsetting operations.
    let ptr = &5;
    let ptr = ptr as *const i32;
    let _val = ptr.wrapping_offset(0);
    let _val = unsafe { ptr.offset(0) };
    let _val = ptr.wrapping_add(0);
    let _val = unsafe { ptr.add(0) };
    let _val = ptr.wrapping_sub(0);
    let _val = unsafe { ptr.sub(0) };
    let _val = unsafe { ptr.offset_from(ptr) };
    let _val = unsafe { ptr.offset_from_unsigned(ptr) };
}

fn test_offset_from() {
    unsafe {
        let buf = [0u32; 4];

        let x = buf.as_ptr() as *const u8;
        let y = x.offset(12);

        assert_eq!(y.offset_from(x), 12);
        assert_eq!(y.offset_from_unsigned(x), 12);
        assert_eq!(x.offset_from(y), -12);
        assert_eq!((y as *const u32).offset_from(x as *const u32), 12 / 4);
        assert_eq!((x as *const u32).offset_from(y as *const u32), -12 / 4);

        let x = (((x as usize) * 2) / 2) as *const u8;
        assert_eq!(y.offset_from(x), 12);
        assert_eq!(y.offset_from_unsigned(x), 12);
        assert_eq!(x.offset_from(y), -12);
    }
}

// This also internally uses offset_from.
fn test_vec_into_iter() {
    let v = Vec::<i32>::new();
    let i = v.into_iter();
    i.size_hint();
}

fn ptr_arith_offset() {
    let v = [1i16, 2];
    let x = &v as *const [i16] as *const i16;
    let x = x.wrapping_offset(1);
    assert_eq!(unsafe { *x }, 2);
}

fn ptr_arith_offset_overflow() {
    let v = [1i16, 2];
    let x = &mut ptr::null(); // going through memory as there are more sanity checks along that path
    *x = v.as_ptr().wrapping_offset(1); // ptr to the 2nd element
    // Adding 2*isize::max and then 1 is like subtracting 1
    *x = x.wrapping_offset(isize::MAX);
    *x = x.wrapping_offset(isize::MAX);
    *x = x.wrapping_offset(1);
    assert_eq!(unsafe { **x }, 1);
}

fn ptr_offset() {
    fn f() -> i32 {
        42
    }

    let v = [1i16, 2];
    let x = &v as *const [i16; 2] as *const i16;
    let x = unsafe { x.offset(1) };
    assert_eq!(unsafe { *x }, 2);

    // fn ptr offset
    unsafe {
        let p = f as fn() -> i32 as usize;
        let x = (p as *mut u32).offset(0) as usize;
        // *cast* to ptr, then transmute to fn ptr.
        // (transmuting int to [fn]ptr causes trouble.)
        let f: fn() -> i32 = mem::transmute(x as *const ());
        assert_eq!(f(), 42);
    }
}
