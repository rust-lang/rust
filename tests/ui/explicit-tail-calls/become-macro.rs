//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls, decl_macro)]

macro call($f:expr $(, $args:expr)* $(,)?) {
    ($f)($($args),*)
}

fn main() {
    become call!(f);
}

fn f() {}
