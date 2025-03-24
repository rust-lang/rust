//@ run-pass

#![allow(unnecessary_refs)]

pub fn main() {
    let x: isize = 3;
    println!("&x={:x}", (&x as *const isize as usize));
}
