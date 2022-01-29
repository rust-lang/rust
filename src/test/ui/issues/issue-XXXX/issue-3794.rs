// run-pass
#![feature(box_syntax)]
#![allow(dead_code)]

trait T {
    fn print(&self);
}

#[derive(Debug)]
struct S {
    s: isize,
}

impl T for S {
    fn print(&self) {
        println!("{:?}", self);
    }
}

fn print_t(t: &dyn T) {
    t.print();
}

fn print_s(s: &S) {
    s.print();
}

pub fn main() {
    let s: Box<S> = box S { s: 5 };
    print_s(&*s);
    let t: Box<dyn T> = s as Box<dyn T>;
    print_t(&*t);
}
