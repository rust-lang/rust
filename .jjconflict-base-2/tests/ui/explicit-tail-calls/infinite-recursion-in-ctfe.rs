#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

const _: () = f();

const fn f() {
    become f(); //~ error: constant evaluation is taking a long time
}

fn main() {}
