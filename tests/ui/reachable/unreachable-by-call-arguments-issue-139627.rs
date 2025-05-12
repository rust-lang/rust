//@ check-pass
#![deny(unreachable_code)]
#![deny(unused)]

pub enum Void {}

pub struct S<T>(T);

pub fn foo(void: Void, void1: Void) {
    let s = S(void);
    drop(s);
    let s1 = S { 0: void1 };
    drop(s1);
}

fn main() {}
