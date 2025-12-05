use byteorder::{BigEndian, ByteOrder};

pub fn use_the_dependency() {
    let _n = <BigEndian as ByteOrder>::read_u32(&[1, 2, 3, 4]);
}
