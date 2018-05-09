// This should fail even without validation
// ignore-test causes rustc ICE: rust-lang/rust#50570
// compile-flags: -Zmir-emit-validate=0

#![feature(never_type)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

struct Human;

fn main() {
    let x: ! = unsafe {
        std::mem::transmute::<Human, !>(Human) //~ ERROR entered unreachable code
    };
    f(x)
}

fn f(x: !) -> ! { x }
