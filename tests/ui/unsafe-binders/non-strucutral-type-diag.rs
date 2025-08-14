// regression test for <https://github.com/rust-lang/rust/issues/141422>.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

#[derive(Copy, Clone)]
struct Adt<'a>(&'a ());

const C: Option<(unsafe<'a> Adt<'a>, Box<dyn Send>)> = None;

fn main() {
    match None {
        C => {}
        //~^ ERROR constant of non-structural type
        _ => {}
    }
}
