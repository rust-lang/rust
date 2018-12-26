// run-pass
#![allow(dead_code)]

trait A { fn a(&self) -> isize; }
trait B: A { fn b(&self) -> isize; }
trait C: A { fn c(&self) -> isize; }

struct S { bogus: () }

impl A for S { fn a(&self) -> isize { 10 } }
impl B for S { fn b(&self) -> isize { 20 } }
impl C for S { fn c(&self) -> isize { 30 } }

// Multiple type params, multiple levels of inheritance
fn f<X:A,Y:B,Z:C>(x: &X, y: &Y, z: &Z) {
    assert_eq!(x.a(), 10);
    assert_eq!(y.a(), 10);
    assert_eq!(y.b(), 20);
    assert_eq!(z.a(), 10);
    assert_eq!(z.c(), 30);
}

pub fn main() {
    let s = &S { bogus: () };
    f(s, s, s);
}
