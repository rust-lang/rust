// compile-flags:-W err-ctypes

#[warn(no_ctypes)];

#[nolink]
native mod libc {
    fn malloc(size: int) -> *u8;
}

fn main() {
}