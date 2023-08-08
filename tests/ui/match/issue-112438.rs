// run-pass
#![feature(inline_const_pat)]
#![allow(dead_code)]
#![allow(incomplete_features)]
fn foo<const V: usize>() {
    match 0 {
        const { 1 << 5 } | _ => {}
    }
}

fn main() {}
