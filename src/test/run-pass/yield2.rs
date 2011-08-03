// -*- rust -*-
use std;

fn main() {
    let i: int = 0;
    while i < 100 { i = i + 1; log_err i; std::task::yield(); }
}