// xfail-win32
use std;
import std::task;
import std::comm;

resource complainer(c: @int) {
}

fn f() {
    task::unsupervise();
    let c <- complainer(@0);
    fail;
}

fn main() {
    let g = f;
    task::spawn(g);
}