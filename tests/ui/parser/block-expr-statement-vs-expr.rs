//! Regression test for <https://github.com/rust-lang/rust/issues/29071>.
//@ run-pass

#![allow(non_upper_case_globals, dead_code)]

fn ret() -> u32 {
    static x: u32 = 10;
    x & if true { 10u32 } else { 20u32 } & x
}

fn ret2() -> &'static u32 {
    static x: u32 = 10;
    if true { 10u32; } else { 20u32; }
    &x
}

fn t1() -> u32 {
    let x;
    x = if true { [1, 2, 3] } else { [2, 3, 4] }[0];
    x
}

fn t2() -> [u32; 1] {
    if true { [1, 2, 3]; } else { [2, 3, 4]; }
    [0]
}

fn t3() -> u32 {
    let x;
    x = if true { i1 as F } else { i2 as F }();
    x
}

fn t4() -> () {
    if true { i1 as F; } else { i2 as F; }
    ()
}

type F = fn() -> u32;
fn i1() -> u32 { 1 }
fn i2() -> u32 { 2 }

fn main() {
    assert_eq!(t1(), 1);
    assert_eq!(t3(), 1);
}
