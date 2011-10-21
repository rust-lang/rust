// xfail-win32
use std;
import std::task;
import std::comm;

resource complainer(c: @int) {
}

fn f(&&_i: ()) {
    task::unsupervise();
    let c <- complainer(@0);
    fail;
}

fn main() {
    task::spawn((), f);
}