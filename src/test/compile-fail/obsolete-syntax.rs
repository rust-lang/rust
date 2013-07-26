// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f1<T:copy>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T:send>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T:const>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T:owned>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

struct s {
    let foo: (),
    //~^ ERROR obsolete syntax: `let` in field declaration
    bar: ();
    //~^ ERROR obsolete syntax: field declaration terminated with semicolon
    new() { }
    //~^ ERROR obsolete syntax: struct constructor
}

struct q : r {
    //~^ ERROR obsolete syntax: class traits
    foo: int
}

struct sss {
    bar: int,
    priv {
    //~^ ERROR obsolete syntax: private section
        foo: ()
    }
}

fn obsolete_with() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: () with a };
    //~^ ERROR obsolete syntax: with
    let c = S { foo: (), with a };
    //~^ ERROR obsolete syntax: with
}

fn obsolete_moves() {
    let mut x = 0;
    let y <- x;
    //~^ ERROR obsolete syntax: initializer-by-move
    y <- x;
    //~^ ERROR obsolete syntax: binary move
}

extern mod obsolete_name {
    //~^ ERROR obsolete syntax: named external module
    fn bar();
}

pub extern {
    //~^ ERROR obsolete syntax: `pub extern`
    pub fn bar();
}

fn main() { }
