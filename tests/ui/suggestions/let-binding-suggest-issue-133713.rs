//@ run-rustfix
#![allow(dead_code)]

fn demo1() {
    let _last = u64 = 0; //~ ERROR expected value, found builtin type `u64`
}

fn demo2() {
    let _val = u64; //~ ERROR expected value, found builtin type `u64`
}

fn main() {}
