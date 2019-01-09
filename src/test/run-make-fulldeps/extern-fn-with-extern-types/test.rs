#![feature(extern_types)]

#[link(name = "ctest", kind = "static")]
extern {
    type data;

    fn data_create(magic: u32) -> *mut data;
    fn data_get(data: *mut data) -> u32;
}

const MAGIC: u32 = 0xdeadbeef;
fn main() {
    unsafe {
        let data = data_create(MAGIC);
        assert_eq!(data_get(data), MAGIC);
    }
}
