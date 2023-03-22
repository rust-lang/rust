// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(const_trait_impl)]

fn foo(_: impl std::marker::Destruct) {}

struct MyAdt;

fn main() {
    foo(1);
    foo(MyAdt);
}
