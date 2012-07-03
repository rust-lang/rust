// This creates a bunch of yielding tasks that run concurrently
// while holding onto C stacks

extern mod rustrt {
    fn rust_dbg_call(cb: *u8,
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

crust fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        task::yield();
        count(data - 1u) + count(data - 1u)
    }
}

fn count(n: uint) -> uint {
    rustrt::rust_dbg_call(cb, n)
}

fn main() {
    do iter::repeat(100u) || {
        do task::spawn || {
            assert count(5u) == 16u;
        };
    }
}