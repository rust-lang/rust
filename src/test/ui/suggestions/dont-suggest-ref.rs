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

enum Either {
    One(X),
    Two(X),
}

struct X(Y);

struct Y;

pub fn main() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let r = &e;
    let rm = &mut Either::One(X(Y));

    let x = X(Y);
    let mut xm = X(Y);

    let s = &x;
    let sm = &mut X(Y);

    // --------

    let X(_t) = *s;
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION s
    if let Either::One(_t) = *r { }
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION r
    while let Either::One(_t) = *r { }
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION r
    match *r {
        //~^ ERROR cannot move
        //~| HELP consider removing this dereference operator
        //~| SUGGESTION r
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match *r {
        //~^ ERROR cannot move
        //~| HELP consider removing this dereference operator
        //~| SUGGESTION r
        // (invalid but acceptable)
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
    }

    let X(_t) = *sm;
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION sm
    if let Either::One(_t) = *rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION rm
    while let Either::One(_t) = *rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing this dereference operator
    //~| SUGGESTION rm
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing this dereference operator
        //~| SUGGESTION rm
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing this dereference operator
        //~| SUGGESTION rm
        // (invalid but acceptable)
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
    }
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing this dereference operator
        //~| SUGGESTION rm
        // (invalid but acceptable)
        Either::One(_t) => (),
        Either::Two(ref mut _t) => (),
    }

    // --------

    let &X(_t) = s;
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)
    if let &Either::One(_t) = r { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    while let &Either::One(_t) = r { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    match r {
        //~^ ERROR cannot move
        &Either::One(_t)
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        | &Either::Two(_t) => (),
        // TODO: would really like a suggestion here too
    }
    match r {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &Either::Two(ref _t) => (),
    }
    match r {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        Either::Two(_t) => (),
    }
    fn f1(&X(_t): &X) { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)

    let &mut X(_t) = sm;
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)
    if let &mut Either::One(_t) = rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    while let &mut Either::One(_t) = rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &mut Either::Two(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::Two(_t)
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &mut Either::Two(ref _t) => (),
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &mut Either::Two(ref mut _t) => (),
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        Either::Two(_t) => (),
    }
    fn f2(&mut X(_t): &mut X) { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)

    // --------

    let &X(_t) = &x;
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)
    if let &Either::One(_t) = &e { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    while let &Either::One(_t) = &e { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t)
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        | &Either::Two(_t) => (),
        // TODO: would really like a suggestion here too
    }
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &Either::Two(ref _t) => (),
    }
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        Either::Two(_t) => (),
    }

    let &mut X(_t) = &mut xm;
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION X(_t)
    if let &mut Either::One(_t) = &mut em { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    while let &mut Either::One(_t) = &mut em { }
    //~^ ERROR cannot move
    //~| HELP consider removing this borrow operator
    //~| SUGGESTION Either::One(_t)
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t)
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        | &mut Either::Two(_t) => (),
        // TODO: would really like a suggestion here too
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &mut Either::Two(ref _t) => (),
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        &mut Either::Two(ref mut _t) => (),
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing this borrow operator
        //~| SUGGESTION Either::One(_t)
        Either::Two(_t) => (),
    }
}
