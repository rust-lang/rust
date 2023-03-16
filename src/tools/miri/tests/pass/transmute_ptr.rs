//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(strict_provenance)]
use std::{mem, ptr};

fn t1() {
    // If we are careful, we can exploit data layout...
    // This is a tricky case since we are transmuting a ScalarPair type to a non-ScalarPair type.
    let raw = unsafe { mem::transmute::<&[u8], [*const u8; 2]>(&[42]) };
    let ptr: *const u8 = unsafe { mem::transmute_copy(&raw) };
    assert_eq!(unsafe { *ptr }, 42);
}

#[cfg(target_pointer_width = "64")]
const PTR_SIZE: usize = 8;
#[cfg(target_pointer_width = "32")]
const PTR_SIZE: usize = 4;

fn t2() {
    let bad = unsafe { mem::transmute::<&[u8], [u8; 2 * PTR_SIZE]>(&[1u8]) };
    let _val = bad[0] + bad[bad.len() - 1];
}

fn ptr_integer_array() {
    let r = &mut 42;
    let _i: [usize; 1] = unsafe { mem::transmute(r) };

    let _x: [u8; PTR_SIZE] = unsafe { mem::transmute(&0) };
}

fn ptr_in_two_halves() {
    unsafe {
        let ptr = &0 as *const i32;
        let arr = [ptr; 2];
        // We want to do a scalar read of a pointer at offset PTR_SIZE/2 into this array. But we
        // cannot use a packed struct or `read_unaligned`, as those use the memcpy code path in
        // Miri. So instead we shift the entire array by a bit and then the actual read we want to
        // do is perfectly aligned.
        let mut target_arr = [ptr::null::<i32>(); 3];
        let target = target_arr.as_mut_ptr().cast::<u8>();
        target.add(PTR_SIZE / 2).cast::<[*const i32; 2]>().write_unaligned(arr);
        // Now target_arr[1] is a mix of the two `ptr` we had stored in `arr`.
        let strange_ptr = target_arr[1];
        // Check that the provenance works out.
        assert_eq!(*strange_ptr.with_addr(ptr.addr()), 0);
    }
}

fn main() {
    t1();
    t2();
    ptr_integer_array();
    ptr_in_two_halves();
}
