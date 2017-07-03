#![feature(core_intrinsics)]

fn main() {
    let x: Option<Box<[u8]>> = unsafe {
        let z = std::intrinsics::add_with_overflow(0usize, 0usize);
        std::mem::transmute::<(usize, bool), Option<Box<[u8]>>>(z)
    };
    let y = &x;
    // Now read this bytewise.  There should be 9 def bytes followed by 7 undef bytes (the padding after the bool) in there.
    let z : *const u8 = y as *const _ as *const _;
    for i in 0..9 {
        let byte = unsafe { *z.offset(i) };
        assert_eq!(byte, 0);
    }
    let v = unsafe { *z.offset(9) };
    if v == 0 {} //~ ERROR attempted to read undefined bytes
}
