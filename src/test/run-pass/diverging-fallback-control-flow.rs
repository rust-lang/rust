// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test various cases where we permit an unconstrained variable
// to fallback based on control-flow.
//
// These represent current behavior, but are pretty dubious.  I would
// like to revisit these and potentially change them. --nmatsakis

#![feature(never_type)]

trait BadDefault {
    fn default() -> Self;
}

impl BadDefault for u32 {
    fn default() -> Self {
        0
    }
}

impl BadDefault for ! {
    fn default() -> ! {
        panic!()
    }
}

fn assignment() {
    let x;

    if true {
        x = BadDefault::default();
    } else {
        x = return;
    }
}

fn assignment_rev() {
    let x;

    if true {
        x = return;
    } else {
        x = BadDefault::default();
    }
}

fn if_then_else() {
    let _x = if true {
        BadDefault::default()
    } else {
        return;
    };
}

fn if_then_else_rev() {
    let _x = if true {
        return;
    } else {
        BadDefault::default()
    };
}

fn match_arm() {
    let _x = match Ok(BadDefault::default()) {
        Ok(v) => v,
        Err(()) => return,
    };
}

fn match_arm_rev() {
    let _x = match Ok(BadDefault::default()) {
        Err(()) => return,
        Ok(v) => v,
    };
}

fn loop_break() {
    let _x = loop {
        if false {
            break return;
        } else {
            break BadDefault::default();
        }
    };
}

fn loop_break_rev() {
    let _x = loop {
        if false {
            break return;
        } else {
            break BadDefault::default();
        }
    };
}

fn main() { }
