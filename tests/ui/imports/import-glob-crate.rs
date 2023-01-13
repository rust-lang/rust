// run-pass
use std::mem::*;

pub fn main() {
    assert_eq!(size_of::<u8>(), 1);
    let (mut x, mut y) = (1, 2);
    swap(&mut x, &mut y);
    assert_eq!(x, 2);
    assert_eq!(y, 1);
}

#[allow(unused)]
fn f() {
    mod foo { pub use *; }
    mod bar { pub use ::*; }

    foo::main();
    bar::main();
}
