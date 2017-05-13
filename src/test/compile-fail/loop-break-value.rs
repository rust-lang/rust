// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(loop_break_value)]
#![feature(never_type)]

fn main() {
    let val: ! = loop { break break; };
    //~^ ERROR mismatched types

    loop {
        if true {
            break "asdf";
        } else {
            break 123; //~ ERROR mismatched types
        }
    };

    let _: i32 = loop {
        break "asdf"; //~ ERROR mismatched types
    };

    let _: i32 = 'outer_loop: loop {
        loop {
            break 'outer_loop "nope"; //~ ERROR mismatched types
            break "ok";
        };
    };

    'while_loop: while true {
        break;
        break (); //~ ERROR `break` with value from a `while` loop
        loop {
            break 'while_loop 123;
            //~^ ERROR `break` with value from a `while` loop
            //~| ERROR mismatched types
            break 456;
            break 789;
        };
    }

    'while_let_loop: while let Some(_) = Some(()) {
        if break () { //~ ERROR `break` with value from a `while let` loop
            break;
            break None;
            //~^ ERROR `break` with value from a `while let` loop
            //~| ERROR mismatched types
        }
        loop {
            break 'while_let_loop "nope";
            //~^ ERROR `break` with value from a `while let` loop
            //~| ERROR mismatched types
            break 33;
        };
    }

    'for_loop: for _ in &[1,2,3] {
        break (); //~ ERROR `break` with value from a `for` loop
        break [()];
        //~^ ERROR `break` with value from a `for` loop
        //~| ERROR mismatched types
        loop {
            break Some(3);
            break 'for_loop Some(17);
            //~^ ERROR `break` with value from a `for` loop
            //~| ERROR mismatched types
        };
    }

    let _: i32 = 'a: loop {
        let _: () = 'b: loop {
            break ('c: loop {
                break;
                break 'c 123; //~ ERROR mismatched types
            });
            break 'a 123;
        };
    };

    loop {
        break (break, break); //~ ERROR mismatched types
    };

    loop {
        break;
        break 2; //~ ERROR mismatched types
    };

    loop {
        break 2;
        break; //~ ERROR mismatched types
        break 4;
    };
}
