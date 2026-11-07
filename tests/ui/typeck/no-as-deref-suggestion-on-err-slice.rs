// Regression test for https://github.com/rust-lang/rust/issues/165365

use std::ops::Deref;

enum E {
    A { x: u8 },
}

fn err_vec(err: Result<(), Vec<E>>) {
    match err {
        Err([E::A { .. }]) => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }
}

fn err_vec_when_ok_derefs(err: Result<String, Vec<u8>>) {
    match err {
        Err([1]) => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }
}

fn err_same_type(err: Result<Vec<E>, Vec<E>>) {
    match err {
        Err([]) => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }
}

fn ok_same_type(err: Result<Vec<E>, Vec<E>>) {
    match err {
        //~^ HELP: consider using `as_deref` here
        Ok([]) => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }
}

enum Contrived<'a> {
    Layer(&'a Contrived<'a>),
    Value(Vec<E>),
}

impl Deref for Contrived<'_> {
    type Target = [E];
    fn deref(&self) -> &[E] {
        match self {
            Contrived::Layer(peeled) => peeled.deref(),
            Contrived::Value(value) => value.deref(),
        }
    }
}

fn contrived(opt: Option<Contrived<'_>>) {
    match opt {
        Some(Contrived::Layer([])) => {}
        //~^ ERROR: expected an array or slice
        _ => {}
    }
}

fn main() {}
