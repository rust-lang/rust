// gate-test-edition_redirect

#![feature(rustc_attrs)]

struct Old;

#[rustc_edition_redirect(before = "2024", target(Old))]
//~^ ERROR the `rustc_edition_redirect` attribute is an experimental feature
struct Current;

fn main() {}
