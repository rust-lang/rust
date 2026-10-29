//! regression test for <https://github.com/rust-lang/rust/issues/50600>
struct Foo(
    fn([u8; |x: u8| {}]), //~ ERROR mismatched types
);

fn main() {}
