// This test will call __morestack with various minimum stack sizes

use std;
import task;

native mod rustrt {
    fn set_min_stack(size: uint);
}

fn getbig(&&i: int) {
    if i != 0 {
        getbig(i - 1);
    }
}

fn main() {
    let sz = 400u;
    while sz < 500u {
        rustrt::set_min_stack(sz);
        task::join(task::spawn_joinable(200, getbig));
        sz += 1u;
    }
}