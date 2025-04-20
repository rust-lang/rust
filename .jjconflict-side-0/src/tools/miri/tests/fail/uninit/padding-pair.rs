//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

#![feature(core_intrinsics)]

use std::mem::{self, MaybeUninit};

fn main() {
    // This constructs a `(usize, bool)` pair: 9 bytes initialized, the rest not.
    // Ensure that these 9 bytes are indeed initialized, and the rest is indeed not.
    // This should be the case even if we write into previously initialized storage.
    let mut x: MaybeUninit<Box<[u8]>> = MaybeUninit::zeroed();
    let z = std::intrinsics::add_with_overflow(0usize, 0usize);
    unsafe { x.as_mut_ptr().cast::<(usize, bool)>().write(z) };
    // Now read this bytewise. There should be (`ptr_size + 1`) def bytes followed by
    // (`ptr_size - 1`) undef bytes (the padding after the bool) in there.
    let z: *const u8 = &x as *const _ as *const _;
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
