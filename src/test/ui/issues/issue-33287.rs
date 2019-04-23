// compile-pass
#![allow(dead_code)]
#![allow(unused_variables)]
const A: [u32; 1] = [0];

fn test() {
    let range = A[1]..;
}

fn main() { }
