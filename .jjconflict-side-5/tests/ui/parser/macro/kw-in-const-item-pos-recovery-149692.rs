//! More test coverage for <https://github.com/rust-lang/rust/issues/149692>; this test is
//! specifically for `const` items.

macro_rules! m {
    (const $id:item()) => {}
}

m!(const Self());
//~^ ERROR expected identifier, found keyword `Self`
//~^^ ERROR missing `fn` or `struct` for function or struct definition

fn main() {}
