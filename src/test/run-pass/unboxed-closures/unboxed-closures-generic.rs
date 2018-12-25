// run-pass
use std::ops::FnMut;

fn call_it<F:FnMut(i32,i32)->i32>(y: i32, mut f: F) -> i32 {
    f(2, y)
}

pub fn main() {
    let f = |x: i32, y: i32| -> i32 { x + y };
    let z = call_it(3, f);
    println!("{}", z);
    assert_eq!(z, 5);
}
