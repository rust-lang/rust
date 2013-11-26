// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn inty(fun: proc(int) -> int) -> int {
    fun(100)
}

fn booly(fun: proc(bool) -> bool) -> bool {
    fun(true)
}

// Check usage and precedence of block arguments in expressions:
pub fn main() {
    let v = ~[-1.0f64, 0.0, 1.0, 2.0, 3.0];

    // Statement form does not require parentheses:
    for i in v.iter() {
        info!("{:?}", *i);
    }

    // Usable at all:
    do inty |x| { x };

    // Higher precedence than assignments:
    let result = do inty |e| { e };
    assert_eq!(result, 100);

    // Higher precedence than unary operations:
    let stringy = do inty |e| { e }.to_str();
    assert!(do booly |_| { true });
    assert!(!do booly |_| { false });

    // Usable in funny statement-like forms:
    if !do booly |_| { true } {
        assert!(false);
    }
    match do booly |_| { false } {
        true => { fail!("incorrect answer."); }
        false => { }
    }
    match 3 {
      _ if do booly |_| { true } => {
      }
      _ => {
        fail!("wrong answer.");
      }
    }


    // Lower precedence than binary operations:
    let w = do inty |_| { 10 } + 10;
    let y = do inty |_| { 10 } + 10;
    let z = 10 + do inty |_| { 10 };
    assert_eq!(w, y);
    assert_eq!(y, z);

    // In the tail of a block
    let w = if true {
        do booly |_| {
            true
        }
    } else {
        false
    };
    assert!(w);
}
