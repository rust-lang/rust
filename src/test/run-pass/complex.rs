#![allow(unconditional_recursion)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused_mut)]



type t = isize;

fn nothing() { }

fn putstr(_s: String) { }

fn putint(_i: isize) {
    let mut i: isize = 33;
    while i < 36 { putstr("hi".to_string()); i = i + 1; }
}

fn zerg(i: isize) -> isize { return i; }

fn foo(x: isize) -> isize {
    let mut y: t = x + 2;
    putstr("hello".to_string());
    while y < 10 { putint(y); if y * 3 == 4 { y = y + 2; nothing(); } }
    let mut z: t;
    z = 0x55;
    foo(z);
    return 0;
}

pub fn main() {
    let x: isize = 2 + 2;
    println!("{}", x);
    println!("hello, world");
    println!("{}", 10);
}
