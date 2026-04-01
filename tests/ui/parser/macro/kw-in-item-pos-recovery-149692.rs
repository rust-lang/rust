//! Regression test for a diagnostic ICE where we tried to recover a keyword as the identifier when
//! we are already trying to recover a missing keyword before item.
//!
//! See <https://github.com/rust-lang/rust/issues/149692>.

macro_rules! m {
    ($id:item()) => {}
}

m!(Self());
//~^ ERROR expected identifier, found keyword `Self`
//~^^ ERROR missing `fn` or `struct` for function or struct definition

m!(Self{});
//~^ ERROR expected identifier, found keyword `Self`
//~^^ ERROR missing `enum` or `struct` for enum or struct definition

m!(crate());
//~^ ERROR expected identifier, found keyword `crate`
//~^^ ERROR missing `fn` or `struct` for function or struct definition

fn main() {}
