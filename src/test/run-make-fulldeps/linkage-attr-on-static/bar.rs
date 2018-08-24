#![feature(linkage)]

#[no_mangle]
#[linkage = "external"]
static BAZ: i32 = 21;

#[link(name = "foo", kind = "static")]
extern {
    fn what() -> i32;
}

fn main() {
    unsafe {
        assert_eq!(what(), BAZ);
    }
}
