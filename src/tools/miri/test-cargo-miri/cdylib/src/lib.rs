use byteorder::{BigEndian, ByteOrder};

#[no_mangle]
extern "C" fn use_the_dependency() {
    let _n = <BigEndian as ByteOrder>::read_u64(&[1, 2, 3, 4, 5, 6, 7, 8]);
}
