// This test will call __morestack with various minimum stack sizes

use std;
import task;

fn getbig(&&i: int) {
    if i != 0 {
        getbig(i - 1);
    }
}

fn main() {
    let sz = 400u;
    while sz < 500u {
        task::join(task::spawn_joinable {|| getbig(200) });
        sz += 1u;
    }
}