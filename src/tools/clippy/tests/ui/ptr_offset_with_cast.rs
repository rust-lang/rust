// run-rustfix

fn main() {
    let vec = vec![b'a', b'b', b'c'];
    let ptr = vec.as_ptr();

    let offset_u8 = 1_u8;
    let offset_usize = 1_usize;
    let offset_isize = 1_isize;

    unsafe {
        ptr.offset(offset_usize as isize);
        ptr.offset(offset_isize as isize);
        ptr.offset(offset_u8 as isize);

        ptr.wrapping_offset(offset_usize as isize);
        ptr.wrapping_offset(offset_isize as isize);
        ptr.wrapping_offset(offset_u8 as isize);
    }
}
