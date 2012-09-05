use std;
use std::arena;

fn main() {
    let p = &arena::Arena();
    let x = p.alloc(|| 4u);
    io::print(fmt!("%u", *x));
    assert *x == 4u;
}
