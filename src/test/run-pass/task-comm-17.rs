// Issue #922

use std;
import std::task;

fn f() {
}

fn main() {
    task::spawn(bind f());
}