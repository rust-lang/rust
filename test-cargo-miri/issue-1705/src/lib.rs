use byteorder::{LittleEndian, ByteOrder};

pub fn use_the_dependency() {
    let _n = <LittleEndian as ByteOrder>::read_u32(&[1,2,3,4]);
}
