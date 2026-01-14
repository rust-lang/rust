//! Regression test for a diagnostic ICE where we tried to recover a keyword as the identifier when
//! we are already trying to recover a missing keyword before item.
//!
//! See <https://github.com/rust-lang/rust/issues/149692>.

macro_rules! m {
    ($id:item()) => {}
}

m!(Self());
//~^ ERROR expected one of `!` or `::`, found `(`

m!(Self{});
//~^ ERROR expected one of `!` or `::`, found `{`

m!(crate());
//~^ ERROR expected one of `!` or `::`, found `(`

fn main() {}
