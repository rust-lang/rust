// error-pattern:explicit failure

// Just testing unwinding

use std;

native mod rustrt {
    fn set_min_stack(size: uint);
}

fn getbig_and_fail(&&i: int) {
    let r = and_then_get_big_again(@0);
    if i != 0 {
        getbig_and_fail(i - 1);
    } else {
        fail;
    }
}

resource and_then_get_big_again(_i: @int) {
}

fn main() {
    rustrt::set_min_stack(256u);
    task::spawn(1, getbig_and_fail);
}