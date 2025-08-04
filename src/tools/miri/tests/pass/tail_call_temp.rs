#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

struct Wrapper(#[expect(unused)] usize);

fn f(t: bool, x: Wrapper) {
    if t { become f(false, x); }
}

fn main() {
    f(true, Wrapper(5));
}
