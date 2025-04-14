#![deny(unreachable_code)]
#![deny(unused)]

pub enum Void {}

pub struct S<T>(T);

pub fn foo(void: Void, void1: Void) { //~ ERROR unused variable: `void1`
    let s = S(void); //~ ERROR unused variable: `s`
    drop(s); //~ ERROR unreachable expression
    let s1 = S { 0: void1 };
    drop(s1);
}

fn main() {}
