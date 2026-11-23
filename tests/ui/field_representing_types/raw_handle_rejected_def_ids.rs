//@ edition: 2024

#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::raw_handle;

#[derive(Copy, Clone)]
struct S {
    a: i32,
}

fn main() {
    let s = S { a: 1 };
    let _ = raw_handle!((|| {
        use std::io::*;
        let _ = std::io::sink().write(&[]);
        s
    })()
    .a);
    //~^^^^^^ ERROR cannot take address of a temporary [E0745]
}
