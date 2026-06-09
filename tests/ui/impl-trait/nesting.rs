//@ run-pass
#![allow(dead_code)]

fn foo<T>(t: T) -> impl Into<[T; { const FOO: usize = 1; FOO }]> {
    [t]
}

fn bar() -> impl Into<[u8; { const FOO: usize = 1; FOO }]> {
    [99]
}

fn main() {
    println!("{:?}", foo(42).into());
    println!("{:?}", bar().into());
}
