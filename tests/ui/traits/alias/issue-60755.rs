//@ check-pass

#![feature(trait_alias)]

struct MyStruct {}
trait MyFn = Fn(&MyStruct);

fn foo(_: impl MyFn) {}

fn main() {
    foo(|_| {});
}
