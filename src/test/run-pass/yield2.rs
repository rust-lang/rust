// -*- rust -*-
extern mod std;

fn main() {
    let mut i: int = 0;
    while i < 100 { i = i + 1; log(error, i); task::yield(); }
}
