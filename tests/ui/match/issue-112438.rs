//@ run-pass
#![feature(inline_const_pat)]
#![allow(dead_code)]
fn foo<const V: usize>() {
    match 0 {
        const { 1 << 5 } | _ => {}
    }
}

fn main() {}
