// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check usage and precedence of block arguments in expressions:
pub fn main() {
    let v = ~[-1.0f64, 0.0, 1.0, 2.0, 3.0];

    // Statement form does not require parentheses:
    for i in v.iter() {
        info2!("{:?}", *i);
    }

    // Usable at all:
    let mut any_negative = do v.iter().any |e| { e.is_negative() };
    assert!(any_negative);

    // Higher precedence than assignments:
    any_negative = do v.iter().any |e| { e.is_negative() };
    assert!(any_negative);

    // Higher precedence than unary operations:
    let abs_v = do v.iter().map |e| { e.abs() }.collect::<~[f64]>();
    assert!(do abs_v.iter().all |e| { e.is_positive() });
    assert!(!do abs_v.iter().any |e| { e.is_negative() });

    // Usable in funny statement-like forms:
    if !do v.iter().any |e| { e.is_positive() } {
        assert!(false);
    }
    match do v.iter().all |e| { e.is_negative() } {
        true => { fail2!("incorrect answer."); }
        false => { }
    }
    match 3 {
      _ if do v.iter().any |e| { e.is_negative() } => {
      }
      _ => {
        fail2!("wrong answer.");
      }
    }


    // Lower precedence than binary operations:
    let w = do v.iter().fold(0.0) |x, y| { x + *y } + 10.0;
    let y = do v.iter().fold(0.0) |x, y| { x + *y } + 10.0;
    let z = 10.0 + do v.iter().fold(0.0) |x, y| { x + *y };
    assert_eq!(w, y);
    assert_eq!(y, z);

    // In the tail of a block
    let w =
        if true { do abs_v.iter().any |e| { e.is_positive() } }
      else { false };
    assert!(w);
}
