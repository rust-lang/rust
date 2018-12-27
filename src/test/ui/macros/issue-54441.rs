#![feature(macros_in_extern)]

macro_rules! m {
    () => {
        let //~ ERROR expected
    };
}

extern "C" {
    m!();
}

fn main() {}
