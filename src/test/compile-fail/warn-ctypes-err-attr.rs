// error-pattern:found rust type
#[warn(err_ctypes)];

#[nolink]
extern mod libc {
    fn malloc(size: int) -> *u8;
}

fn main() {
}