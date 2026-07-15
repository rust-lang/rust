//@ revisions: edition2018 edition2024
//@[edition2018] edition: 2018
//@[edition2024] edition: 2024
//@ aux-build: macro-source.rs
//@ aux-build: macro-2018.rs
//@ aux-build: macro-2024.rs
//@ check-pass

extern crate macro_2018;
extern crate macro_2024;
extern crate macro_source;

// Redirects in a macro-generated glob use the edition of the glob's path, not
// the edition of the crate where the macro is invoked.
#[cfg(edition2018)]
macro_2024::import_all!();
#[cfg(edition2024)]
macro_2018::import_all!();

fn main() {
    #[cfg(edition2018)]
    let _: macro_source::Current = Name;
    #[cfg(edition2024)]
    let _: macro_source::Old = Name;
}
