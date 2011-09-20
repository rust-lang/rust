// error-pattern:goodfail

use std;
import std::task;
import std::comm;

fn goodfail() {
    task::yield();
    fail "goodfail";
}

fn main() {
    task::spawn(bind goodfail());
    let po = comm::port();
    // We shouldn't be able to get past this recv since there's no
    // message available
    let i: int = comm::recv(po);
    fail "badfail";
}
