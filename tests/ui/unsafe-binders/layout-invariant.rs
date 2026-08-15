// Regression test for https://github.com/rust-lang/rust/issues/154426
#![feature(unsafe_binders)]

#[derive(Copy, Clone)]
struct Adt<'a> {
    a: &'a String,
}

const None: Option<unsafe<> Option<unsafe<'a> Adt<'a>>> = None;
//~^ ERROR the trait bound `unsafe<'a> Adt<'a>: Copy` is not satisfied
//~| ERROR the trait bound `unsafe<'a> Adt<'a>: Copy` is not satisfied

fn main() {
    match None {
        _ => {}
    };
}
