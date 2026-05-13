//@ run-pass
#![allow(unused_variables)]

fn foo<T, F: FnOnce(T) -> T>(f: F) {}
fn id<'a>(input: &'a u8) -> &'a u8 { input }

fn main() {
    foo(id);
}
