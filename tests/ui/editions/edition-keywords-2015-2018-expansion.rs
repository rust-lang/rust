//@ edition:2015
//@ aux-build:edition-kw-macro-2018.rs

#[macro_use]
extern crate edition_kw_macro_2018;

mod one_async {
    produces_async! {} //~ ERROR expected identifier, found keyword
}
mod two_async {
    produces_async_raw! {} // OK
}

fn main() {}
