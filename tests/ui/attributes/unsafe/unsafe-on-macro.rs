#![feature(rustc_attrs)]

macro_rules! bar {
    () => {};
}

#[unsafe(rustc_dummy)]
//~^ ERROR `rustc_dummy` is not an unsafe attribute
bar!();

fn main() {}
