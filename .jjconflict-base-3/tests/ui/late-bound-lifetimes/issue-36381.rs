//@ run-pass
// Regression test for #36381. The monomorphization collector was asserting that
// there are no projection types, but the `<&str as
// StreamOnce>::Position` projection contained a late-bound region,
// and we don't currently normalize in that case until the function is
// actually invoked.

pub trait StreamOnce {
    type Position;
}

impl<'a> StreamOnce for &'a str {
    type Position = usize;
}

pub fn parser<F>(_: F) {
}

fn follow(_: &str) -> <&str as StreamOnce>::Position {
    panic!()
}

fn main() {
    parser(follow);
}
