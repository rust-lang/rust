//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(message = "Self = `{Self}`, \n \
    T = `{T}`, this = `{this}`, path = `{this:path}`, \n \
    resolved = `{this:resolved}`, context = `{item_context}`")]
pub trait From<T>: Sized {
    fn from(x: T) -> Self;
}

fn main() {
    let x: i8 = From::from(42_i32);
}
