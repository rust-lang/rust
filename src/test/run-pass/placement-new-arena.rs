use std;
import std::arena::arena;

fn main() {
    let p = &arena();
    let x = new(*p) 4u;
    io::print(fmt!{"%u", *x});
    assert *x == 4u;
}
