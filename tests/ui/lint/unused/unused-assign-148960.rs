//@ check-fail
#![deny(unused)]
#![allow(dead_code)]

fn test_one_extra_assign() {
    let mut value = b"0".to_vec(); //~ ERROR value assigned to `value` is never read
    value = b"1".to_vec();
    println!("{:?}", value);
}

fn test_two_extra_assign() {
    let mut x = 1; //~ ERROR value assigned to `x` is never read
    x = 2; //~ ERROR value assigned to `x` is never read
    x = 3;
    println!("{}", x);
}

struct Point {
    x: i32,
    y: i32,
}

fn test_indirect_assign() {
    let mut p = Point { x: 1, y: 1 }; //~ ERROR value assigned to `p` is never read
    p = Point { x: 2, y: 2 };
    p.x = 3;
    println!("{}", p.y);
}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

// testcase for issue #148418
fn test_unused_variable() {
    let mut foo = Foo; //~ ERROR variable `foo` is assigned to, but never used
    foo = Foo; //~ ERROR value assigned to `foo` is never read
}

fn main() {}
