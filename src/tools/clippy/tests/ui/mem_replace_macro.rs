// aux-build:macro_rules.rs
#![warn(clippy::mem_replace_with_default)]

#[macro_use]
extern crate macro_rules;

macro_rules! take {
    ($s:expr) => {
        std::mem::replace($s, Default::default())
    };
}

fn replace_with_default() {
    let s = &mut String::from("foo");
    take!(s);
    take_external!(s);
}

fn main() {
    replace_with_default();
}
