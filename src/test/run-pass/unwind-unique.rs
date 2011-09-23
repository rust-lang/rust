// xfail-win32
use std;
import std::task;

fn f() {
    task::unsupervise();
    let a = ~0;
    fail;
}

fn main() {
    let g = f;
    task::spawn(g);
}