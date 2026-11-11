// https://github.com/rust-lang/rust/issues/53419
//@ check-pass

struct Foo {
    bar: dyn for<'r> Fn(usize, &'r dyn FnMut())
}

fn main() {
}
