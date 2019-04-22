// http://rust-lang.org/COPYRIGHT.

#![feature(slice_patterns)]

fn main() {
    let mut a = [1, 2, 3, 4];
    let t = match a {
        [1, 2, ref tail..] => tail,
        _ => unreachable!()
    };
    println!("t[0]: {}", t[0]);
    a[2] = 0; //~ ERROR cannot assign to `a[_]` because it is borrowed
    println!("t[0]: {}", t[0]);
    t[0];
}
