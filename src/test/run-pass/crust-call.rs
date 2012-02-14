native mod rustrt {
    fn rust_dbg_call(cb: *u8,
                     data: ctypes::uintptr_t) -> ctypes::uintptr_t;
}

crust fn cb(data: ctypes::uintptr_t) -> ctypes::uintptr_t {
    if data == 1u {
        data
    } else {
        fact(data - 1u) * data
    }
}

fn fact(n: uint) -> uint {
    #debug("n = %?", n);
    rustrt::rust_dbg_call(cb, n)
}

fn main() {
    let result = fact(10u);
    #debug("result = %?", result);
    assert result == 3628800u;
}