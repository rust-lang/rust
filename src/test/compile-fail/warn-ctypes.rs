// compile-flags:-D ctypes
// error-pattern:found rust type
#[nolink]
extern mod libc {
    #[legacy_exports];
    fn malloc(size: int) -> *u8;
}

fn main() {
}