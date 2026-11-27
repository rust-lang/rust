#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn main() {
    || become f(); //~ error: `become` is not allowed in closures
}

fn f() {}
