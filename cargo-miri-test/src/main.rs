extern crate byteorder;

use byteorder::{BigEndian, ByteOrder};

fn main() {
    let buf = &[1,2,3,4];
    let n = <BigEndian as ByteOrder>::read_u32(buf);
    assert_eq!(n, 0x01020304);
    //println!("{:#x}", n); FIXME enable once memrchr works in miri
    eprintln!("standard error");
}
