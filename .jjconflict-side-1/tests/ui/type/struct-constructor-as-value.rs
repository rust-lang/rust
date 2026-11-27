//! regression test for <https://github.com/rust-lang/rust/issues/35241>

struct Foo(u32);

fn test() -> Foo { Foo } //~ ERROR mismatched types

fn main() {}
