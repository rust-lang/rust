#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn foo() -> for<'a> fn(&'a i32) {
    become bar();
    //~^ ERROR mismatched signatures
}

fn bar() -> fn(&'static i32) {
    dummy
}

fn dummy(_: &i32) {}

fn main() {}
