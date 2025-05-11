//@ run-pass
#![allow(unused_assignments)]
#![allow(unused_variables)]

pub fn main() {
    let mut y: isize = 42;
    let mut z: isize = 42;
    let mut x: isize;
    while z < 50 {
        z += 1;
        while false { x = y; y = z; }
        println!("{}", y);
    }
    assert!(y == 42 && z == 50);
}
