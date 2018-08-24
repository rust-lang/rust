#![allow(warnings)]
#![feature(in_band_lifetimes)]

fn foo(x: &'a u32) -> &'a u32 { x }

fn main() {
    let mut p = 3;
    let r = foo(&p);
    p += 1; //~ ERROR cannot assign to `p` because it is borrowed
    println!("{}", r);
}
