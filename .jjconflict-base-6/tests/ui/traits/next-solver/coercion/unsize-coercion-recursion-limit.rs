//@ check-pass
//@ compile-flags: -Znext-solver

// A test to ensure that unsized coercion is not aborted when visiting a nested goal that
// exceeds the recursion limit and evaluates to `Certainty::Maybe`.
// See https://github.com/rust-lang/rust/pull/152444.

#![allow(warnings)]

struct W<T: ?Sized>(T);
type Four<T: ?Sized> = W<W<W<W<T>>>>;
type Sixteen<T: ?Sized> = Four<Four<Four<Four<T>>>>;

fn ret<T>(x: T) -> Sixteen<T> {
    todo!();
}

fn please_coerce() {
    let mut y = Default::default();
    let x = ret(y);
    let _: &Sixteen<dyn Send> = &x;
    y = 1u32;
}

fn main() {}
