//@ check-pass
//@ run-rustfix

#![warn(unused_visibilities)]

pub const _: () = {};
//~^WARN visibility qualifiers have no effect on `const _` declarations

pub(self) const _: () = {};
//~^WARN visibility qualifiers have no effect on `const _` declarations

fn main() {}
