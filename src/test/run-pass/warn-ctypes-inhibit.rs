// compile-flags:-D ctypes

#[allow(ctypes)];

#[nolink]
extern mod libc {
    fn malloc(size: int) -> *u8;
}

fn main() {
}