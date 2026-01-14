//! More test coverage for <https://github.com/rust-lang/rust/issues/149692>; this test is
//! specifically for `const` items.

macro_rules! m {
    (const $id:item()) => {}
}

m!(const Self());
//~^ ERROR expected one of `!` or `::`, found `(`

fn main() {}
