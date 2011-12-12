// xfail-test
// error-pattern:explicit failure
// compile-flags:--stack-growth

// This time we're testing that the stack limits are restored
// correctly after calling into the C stack and unwinding.
// See the hack in upcall_call_shim_on_c_stack where it messes
// with the stack limit.

use std;

native mod rustrt {
    fn set_min_stack(size: uint);
    fn pin_task();
}

fn getbig_call_c_and_fail(i: int) {
    if i != 0 {
        getbig_call_c_and_fail(i - 1);
    } else {
        rustrt::pin_task();
        fail;
    }
}

resource and_then_get_big_again(_i: ()) {
    fn getbig(i: int) {
        if i != 0 {
            getbig(i - 1);
        }
    }
    getbig(10000);
}

fn main() {
    rustrt::set_min_stack(1024u);
    std::task::spawn((), fn (&&_i: ()) {
        let r = and_then_get_big_again(());
        getbig_call_c_and_fail(10000);
    });
}