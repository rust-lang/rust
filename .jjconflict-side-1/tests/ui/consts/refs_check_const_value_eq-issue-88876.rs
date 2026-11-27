//@ check-pass

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

struct FooConst<const ARRAY: &'static [&'static str]> {}

const FOO_ARR: &[&'static str; 2] = &["Hello", "Friend"];

fn main() {
    let _ = FooConst::<FOO_ARR> {};
}
