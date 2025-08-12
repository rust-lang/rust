// https://github.com/rust-lang/rust/issues/54094
//@ check-pass
trait Zoo {
    type X;
}

impl Zoo for u16 {
    type X = usize;
}

fn foo(abc: <u16 as Zoo>::X) {}

fn main() {
    let x: *const u8 = foo as _;
}
