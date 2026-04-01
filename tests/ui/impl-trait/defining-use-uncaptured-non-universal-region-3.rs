//@ check-pass

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

trait Bar<const FOO: &'static str> {}
impl Bar<"asdf"> for () {}

fn foo<const FOO: &'static str>() -> impl Bar<"asdf"> {
    ()
}

fn main() {}
