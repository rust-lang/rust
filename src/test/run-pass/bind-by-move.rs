// xfail-test
use std;
import std::arc;
fn dispose(+_x: arc::ARC<bool>) unsafe { }

fn main() {
    let p = arc::arc(true);
    let x = some(p);
    match move x {
        some(move z) => { dispose(z); },
        none => fail
    }
}
