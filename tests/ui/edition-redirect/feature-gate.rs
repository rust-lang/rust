// gate-test-edition_redirect

#![feature(rustc_attrs)]

pub struct Old;

#[rustc_edition_redirect = "2024"]
//~^ ERROR the `rustc_edition_redirect` attribute is an experimental feature
pub use Old as Current;

pub struct Current;

fn main() {}
