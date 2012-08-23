extern mod rustrt {
    fn rust_dbg_call(cb: *u8,
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        count(data - 1u) + count(data - 1u)
    }
}

fn count(n: uint) -> uint {
    task::yield();
    rustrt::rust_dbg_call(cb, n)
}

fn main() {
    for iter::repeat(10u) {
        do task::spawn {
            let result = count(5u);
            debug!("result = %?", result);
            assert result == 16u;
        };
    }
}
