// error-pattern:explicit failure
// Testing that runtime failure doesn't cause callbacks to abort abnormally.
// Instead the failure will be delivered after the callbacks return.

extern mod rustrt {
    fn rust_dbg_call(cb: *u8,
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

crust fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
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
    do iter::repeat(10u) || {
        do task::spawn || {
            let result = count(5u);
            #debug("result = %?", result);
            fail;
        };
    }
}
