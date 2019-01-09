use std::ops::FnMut;

pub fn main() {
    let mut f = |x: isize, y: isize| -> isize { x + y };
    let z = f(1_usize, 2);    //~ ERROR mismatched types
    println!("{}", z);
}
