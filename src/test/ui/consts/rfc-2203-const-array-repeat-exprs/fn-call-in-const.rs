// run-pass

#![allow(unused, incomplete_features)]
#![feature(const_in_array_repeat_expressions)]
#![feature(inline_const)]

// Some type that is not copyable.
struct Bar;

const fn type_no_copy() -> Option<Bar> {
    None
}

const fn type_copy() -> u32 {
    3
}

const _: [u32; 2] = [type_copy(); 2];

const _: [Option<Bar>; 2] = [const { type_no_copy() }; 2];

fn main() {}
