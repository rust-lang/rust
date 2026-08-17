//@ revisions: edition2018 edition2024
//@[edition2018] edition: 2018
//@[edition2024] edition: 2024
//@ aux-build: macro-source.rs
//@ aux-build: macro-2018.rs
//@ aux-build: macro-2024.rs
//@ check-pass

extern crate macro_2018;
extern crate macro_2024;
extern crate macro_source as source;

// As with a glob import, importing every macro through `#[macro_use]` uses the
// edition of the generated `extern crate` item.
#[cfg(edition2018)]
macro_2024::macro_use_source!();
#[cfg(edition2024)]
macro_2018::macro_use_source!();

redirected_macro!();

#[cfg(edition2018)]
fn check(value: Selected) -> source::Current {
    value
}

#[cfg(edition2024)]
fn check(value: Selected) -> source::Old {
    value
}

fn main() {}
