// edition:2015
// aux-build:edition-kw-macro-2015.rs
// build-pass (FIXME(62277): could be check-pass?)

#![allow(keyword_idents)]

#[macro_use]
extern crate edition_kw_macro_2015;

mod one_async {
    produces_async! {} // OK
}
mod two_async {
    produces_async_raw! {} // OK
}

fn main() {}
