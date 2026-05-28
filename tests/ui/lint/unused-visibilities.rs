//@ check-pass
//@ run-rustfix

#![warn(unused_visibilities)]

pub const _: () = {};
//~^WARN visibility qualifiers have no effect on `const _` declarations

pub(self) const _: () = {};
//~^WARN visibility qualifiers have no effect on `const _` declarations

macro_rules! foo {
    () => {
        pub const _: () = {};
        //~^WARN visibility qualifiers have no effect on `const _` declarations
    };
}

foo!();

macro_rules! bar {
    ($tt:tt) => {
        pub const $tt: () = {};
    };
}

bar!(_);

fn main() {}
