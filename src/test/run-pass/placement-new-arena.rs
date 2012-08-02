use std;
import std::arena::arena;

fn main() {
    let p = &arena();
    let x = p.alloc(|| 4u);
    io::print(fmt!{"%u", *x});
    assert *x == 4u;
}
