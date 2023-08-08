//@run-rustfix
#![allow(clippy::unnecessary_cast, clippy::useless_vec)]

fn main() {
    let vec = vec![b'a', b'b', b'c'];
    let ptr = vec.as_ptr();

    let offset_u8 = 1_u8;
    let offset_usize = 1_usize;
    let offset_isize = 1_isize;

    unsafe {
        let _ = ptr.offset(offset_usize as isize);
        let _ = ptr.offset(offset_isize as isize);
        let _ = ptr.offset(offset_u8 as isize);

        let _ = ptr.wrapping_offset(offset_usize as isize);
        let _ = ptr.wrapping_offset(offset_isize as isize);
        let _ = ptr.wrapping_offset(offset_u8 as isize);
    }
}
