// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(custom_attribute)]
#![feature(associated_consts)]

macro_rules! stmt_mac {
    () => {
        fn b() {}
    }
}

fn main() {
    #[attr]
    fn a() {}

    #[attr] //~ ERROR 15701
    {

    }

    #[attr] //~ ERROR 15701
    5;

    #[attr] //~ ERROR 15701
    stmt_mac!();
}

// Check that cfg works right

#[cfg(unset)]
fn c() {
    #[attr]
    5;
}

#[cfg(not(unset))]
fn j() {
    #[attr] //~ ERROR 15701
    5;
}

#[cfg_attr(not(unset), cfg(unset))]
fn d() {
    #[attr]
    8;
}

#[cfg_attr(not(unset), cfg(not(unset)))]
fn i() {
    #[attr] //~ ERROR 15701
    8;
}

// check that macro expansion and cfg works right

macro_rules! item_mac {
    ($e:ident) => {
        fn $e() {
            #[attr] //~ ERROR 15701
            42;

            #[cfg(unset)]
            fn f() {
                #[attr]
                5;
            }

            #[cfg(not(unset))]
            fn k() {
                #[attr] //~ ERROR 15701
                5;
            }

            #[cfg_attr(not(unset), cfg(unset))]
            fn g() {
                #[attr]
                8;
            }

            #[cfg_attr(not(unset), cfg(not(unset)))]
            fn h() {
                #[attr] //~ ERROR 15701
                8;
            }

        }
    }
}

item_mac!(e);

// check that the gate visitor works right:

extern {
    #[cfg(unset)]
    fn x(a: [u8; #[attr] 5]);
    fn y(a: [u8; #[attr] 5]); //~ ERROR 15701
}

struct Foo;
impl Foo {
    #[cfg(unset)]
    const X: u8 = #[attr] 5;
    const Y: u8 = #[attr] 5; //~ ERROR 15701
}

trait Bar {
    #[cfg(unset)]
    const X: [u8; #[attr] 5];
    const Y: [u8; #[attr] 5]; //~ ERROR 15701
}

struct Joyce {
    #[cfg(unset)]
    field: [u8; #[attr] 5],
    field2: [u8; #[attr] 5] //~ ERROR 15701
}

struct Walky(
    #[cfg(unset)] [u8; #[attr] 5],
    [u8; #[attr] 5] //~ ERROR 15701
);

enum Mike {
    Happy(
        #[cfg(unset)] [u8; #[attr] 5],
        [u8; #[attr] 5] //~ ERROR 15701
    ),
    Angry {
        #[cfg(unset)]
        field: [u8; #[attr] 5],
        field2: [u8; #[attr] 5] //~ ERROR 15701
    }
}

fn pat() {
    match 5 {
        #[cfg(unset)]
        5 => #[attr] (),
        6 => #[attr] (), //~ ERROR 15701
        _ => (),
    }
}
