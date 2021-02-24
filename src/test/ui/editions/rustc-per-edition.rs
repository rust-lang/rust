// check-pass
// edition:2021
// aux-build:per-edition.rs

#![feature(rustc_attrs)]

#[macro_use]
extern crate per_edition;

#[rustc_per_edition]
type X = (
    u32, // 2015
    &'static str, // 2018,
    f64, // 2021+
);

fn main() {
    let _: X = 6.28;
    let _: per_edition::I32 = 1i32;
    let _: per_edition::I32OrStr = "hello";
    let _: per_edition::Magic = "world";
    let _: per_edition::int!() = 2i32;
    let _: per_edition::x!() = 3u32;
}
