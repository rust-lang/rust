// error-pattern:found rust type
#[warn(err_ctypes)];

#[nolink]
native mod libc {
    fn malloc(size: int) -> *u8;
}

fn main() {
}