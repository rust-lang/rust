//! Regression test for https://github.com/rust-lang/rust/issues/10545

mod a {
    struct S;
    impl S { }
}

fn foo(_: a::S) { //~ ERROR: struct `S` is private
}

fn main() {}
