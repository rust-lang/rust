#![feature(destructuring_assignment)]

const C: i32 = 1;

fn main() {
    let (mut a, mut b);
    (a, .., b, ..) = (0, 1); //~ ERROR `..` can only be used once per tuple pattern
    (a, a, b) = (1, 2); //~ ERROR mismatched types
    (C, ..) = (0,1); //~ ERROR invalid left-hand side of assignment
    (_,) = (1, 2); //~ ERROR mismatched types
}
