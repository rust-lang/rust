//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-permissive-provenance
#![feature(ptr_metadata, const_raw_ptr_comparison)]

use std::mem::{self, transmute};
use std::ptr;

fn one_line_ref() -> i16 {
    *&1
}

fn basic_ref() -> i16 {
    let x = &1;
    *x
}

fn basic_ref_mut() -> i16 {
    let x = &mut 1;
    *x += 2;
    *x
}

fn basic_ref_mut_var() -> i16 {
    let mut a = 1;
    {
        let x = &mut a;
        *x += 2;
    }
    a
}

fn tuple_ref_mut() -> (i8, i8) {
    let mut t = (10, 20);
    {
        let x = &mut t.1;
        *x += 2;
    }
    t
}

fn match_ref_mut() -> i8 {
    let mut t = (20, 22);
    {
        let opt = Some(&mut t);
        match opt {
            Some(&mut (ref mut x, ref mut y)) => *x += *y,
            None => {}
        }
    }
    t.0
}

fn dangling_pointer() -> *const i32 {
    let b = Box::new((42, 42)); // make it bigger than the alignment, so that there is some "room" after this pointer
    &b.0 as *const i32
}

fn wide_ptr_ops() {
    let a: *const dyn Send = &1 as &dyn Send;
    let b: *const dyn Send = &1 as &dyn Send;
    let _val = a == b;
    let _val = a != b;
    let _val = a < b;
    let _val = a <= b;
    let _val = a > b;
    let _val = a >= b;

    let a: *const [u8] = unsafe { transmute((1usize, 1usize)) };
    let b: *const [u8] = unsafe { transmute((1usize, 2usize)) };
    // confirmed with rustc.
    assert!(!(a == b));
    assert!(a != b);
    assert!(a <= b);
    assert!(a < b);
    assert!(!(a >= b));
    assert!(!(a > b));
}

fn metadata_vtable() {
    let p = &0i32 as &dyn std::fmt::Debug;
    let meta: ptr::DynMetadata<_> = ptr::metadata(p as *const _);
    assert_eq!(meta.size_of(), mem::size_of::<i32>());
    assert_eq!(meta.align_of(), mem::align_of::<i32>());

    type T = [i32; 16];
    let p = &T::default() as &dyn std::fmt::Debug;
    let meta: ptr::DynMetadata<_> = ptr::metadata(p as *const _);
    assert_eq!(meta.size_of(), mem::size_of::<T>());
    assert_eq!(meta.align_of(), mem::align_of::<T>());
}

fn main() {
    assert_eq!(one_line_ref(), 1);
    assert_eq!(basic_ref(), 1);
    assert_eq!(basic_ref_mut(), 3);
    assert_eq!(basic_ref_mut_var(), 3);
    assert_eq!(tuple_ref_mut(), (10, 22));
    assert_eq!(match_ref_mut(), 42);

    // Compare even dangling pointers with NULL, and with others in the same allocation, including
    // out-of-bounds.
    assert!(dangling_pointer() != std::ptr::null());
    assert!(match dangling_pointer() as usize {
        0 => false,
        _ => true,
    });
    let dangling = dangling_pointer();
    assert!(dangling == dangling);
    assert!(dangling.wrapping_add(1) != dangling);
    assert!(dangling.wrapping_sub(1) != dangling);

    // Compare pointer with BIG integers
    let dangling = dangling as usize;
    assert!(dangling != usize::MAX);
    assert!(dangling != usize::MAX - 1);
    assert!(dangling != usize::MAX - 2);
    assert!(dangling != usize::MAX - 3); // this is even 4-aligned, but it still cannot be equal because of the extra "room" after this pointer
    assert_eq!((usize::MAX - 3) % 4, 0); // just to be sure we got this right

    // Compare pointer with unaligned integers
    assert!(dangling != 1usize);
    assert!(dangling != 2usize);
    assert!(dangling != 3usize);
    // 4 is a possible choice! So we cannot compare with that.
    assert!(dangling != 5usize);
    assert!(dangling != 6usize);
    assert!(dangling != 7usize);

    // Using inequality to do the comparison.
    assert!(dangling > 0);
    assert!(dangling > 1);
    assert!(dangling > 2);
    assert!(dangling > 3);
    assert!(dangling >= 4);

    // CTFE-specific equality tests, need to also work at runtime.
    let addr = &13 as *const i32;
    let addr2 = (addr as usize).wrapping_add(usize::MAX).wrapping_add(1);
    assert_eq!(addr.guaranteed_eq(addr2 as *const _), Some(true));
    assert_eq!(addr.guaranteed_ne(0x100 as *const _), Some(true));

    wide_ptr_ops();
    metadata_vtable();
}
