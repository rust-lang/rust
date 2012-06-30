// This time we're testing repeatedly going up and down both stacks to
// make sure the stack pointers are maintained properly in both
// directions

native mod rustrt {
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
    #debug("n = %?", n);
    rustrt::rust_dbg_call(cb, n)
}

fn main() {
    // Make sure we're on a task with small Rust stacks (main currently
    // has a large stack)
    do task::spawn || {
        let result = count(12u);
        #debug("result = %?", result);
        assert result == 2048u;
    };
}