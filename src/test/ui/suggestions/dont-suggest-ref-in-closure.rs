// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

#[derive(Clone)]
enum Either {
    One(X),
    Two(X),
}

#[derive(Clone)]
struct X(Y);

#[derive(Clone)]
struct Y;

fn consume_fn<F: Fn()>(_f: F) { }

fn consume_fnmut<F: FnMut()>(_f: F) { }

pub fn main() { }

fn move_into_fn() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // -------- move into Fn --------

    consume_fn(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &x
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &e
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &e
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &e
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &e
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &x
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &em
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &em
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &em
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &em
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}

fn move_into_fnmut() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // -------- move into FnMut --------

    consume_fnmut(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &x
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &e
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &e
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &e
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &e
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &x
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &em
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        //~| SUGGESTION &em
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &em
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &em
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            //~| SUGGESTION &em
            Either::One(mut _t) => (),
            Either::Two(ref mut _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}
