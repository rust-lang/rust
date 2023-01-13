#![feature(core_intrinsics)]

use std::mem;

fn main() {
    let x: Option<Box<[u8]>> = unsafe {
        let z = std::intrinsics::add_with_overflow(0usize, 0usize);
        std::mem::transmute::<(usize, bool), Option<Box<[u8]>>>(z)
    };
    let y = &x;
    // Now read this bytewise. There should be (`ptr_size + 1`) def bytes followed by
    // (`ptr_size - 1`) undef bytes (the padding after the bool) in there.
    let z: *const u8 = y as *const _ as *const _;
    let first_undef = mem::size_of::<usize>() as isize + 1;
    for i in 0..first_undef {
        let byte = unsafe { *z.offset(i) };
        assert_eq!(byte, 0);
    }
    let v = unsafe { *z.offset(first_undef) };
    //~^ ERROR: uninitialized
    if v == 0 {
        println!("it is zero");
    }
}
