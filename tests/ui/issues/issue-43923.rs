//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
struct A<T: ?Sized> { ptr: T }

fn foo<T>(x: &A<[T]>) {}

fn main() {
    let a = foo;
    let b = A { ptr: [a, a, a] };
    a(&A { ptr: [()] });
}
