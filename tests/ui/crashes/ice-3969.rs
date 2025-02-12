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
//~^ ERROR: trait bound

fn unsized_local()
where
    for<'a> Dst<dyn A + 'a>: Sized,
    //~^ ERROR: trait bound
{
    let x: Dst<dyn A> = *(Box::new(Dst { x: 1 }) as Box<Dst<dyn A>>);
}

fn return_str() -> str
where
    str: Sized,
    //~^ ERROR: trait bound
{
    *"Sized".to_string().into_boxed_str()
}

fn use_op(s: String) -> String
where
    String: ::std::ops::Neg<Output = String>,
    //~^ ERROR: trait bound
{
    -s
}

fn use_for()
where
    i32: Iterator,
    //~^ ERROR: trait bound
{
    for _ in 2i32 {}
}

fn main() {}
