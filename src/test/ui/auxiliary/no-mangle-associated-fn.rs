#![crate_type = "lib"]

struct Bar;

impl Bar {
    #[no_mangle]
    fn bar() -> u8 {
        2
    }
}
