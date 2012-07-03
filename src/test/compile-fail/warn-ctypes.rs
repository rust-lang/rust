// compile-flags:-W err-ctypes
// error-pattern:found rust type
#[nolink]
extern mod libc {
    fn malloc(size: int) -> *u8;
}

fn main() {
}