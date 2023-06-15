// run-pass
#![feature(explicit_tail_calls)]

fn main() {
    become f();
}

fn f() {}
