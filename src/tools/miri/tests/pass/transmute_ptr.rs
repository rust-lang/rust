//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::mem;

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

fn main() {
    t1();
    t2();
    ptr_integer_array();
}
