// https://github.com/rust-lang/rust-clippy/issues/3969
// used to crash: error: internal compiler error:
// src/librustc_traits/normalize_erasing_regions.rs:43: could not fully normalize `<i32 as
// std::iter::Iterator>::Item` test from rustc ./ui/trivial-bounds/trivial-bounds-inconsistent.rs

// Check that tautalogically false bounds are accepted, and are used
// in type inference.
#![feature(trivial_bounds)]
#![allow(unused)]
trait A {}

impl A for i32 {}

struct Dst<X: ?Sized> {
    x: X,
}

struct TwoStrs(str, str)
where
    str: Sized;
//~^ ERROR: trait bound str: std::marker::Sized does not depend on any type or lifetime
//~| NOTE: `-D trivial-bounds` implied by `-D warnings`

fn unsized_local()
where
    for<'a> Dst<dyn A + 'a>: Sized,
    //~^ ERROR: trait bound for<'a> Dst<(dyn A + 'a)>: std::marker::Sized does not depend
{
    let x: Dst<dyn A> = *(Box::new(Dst { x: 1 }) as Box<Dst<dyn A>>);
}

fn return_str() -> str
where
    str: Sized,
    //~^ ERROR: trait bound str: std::marker::Sized does not depend on any type or lifetim
{
    *"Sized".to_string().into_boxed_str()
}

fn use_op(s: String) -> String
where
    String: ::std::ops::Neg<Output = String>,
    //~^ ERROR: trait bound std::string::String: std::ops::Neg does not depend on any type
{
    -s
}

fn use_for()
where
    i32: Iterator,
    //~^ ERROR: trait bound i32: std::iter::Iterator does not depend on any type or lifeti
{
    for _ in 2i32 {}
}

fn main() {}
