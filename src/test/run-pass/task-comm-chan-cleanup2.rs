// xfail-win32 Requires unwinding
use std;
import std::comm;

fn main() {
    let p = comm::port();
    let c = comm::chan(p);
    comm::send(c, ~"coffee");
}