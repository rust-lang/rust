// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(asm, macro_rules)]

type History = Vec<&'static str>;

fn wrap<A>(x:A, which: &'static str, history: &mut History) -> A {
    history.push(which);
    x
}

macro_rules! demo {
    ( $output_constraint:tt ) => {
        {
            let mut x: int = 0;
            let y: int = 1;

            let mut history: History = vec!();
            unsafe {
                asm!("mov ($1), $0"
                     : $output_constraint (*wrap(&mut x, "out", &mut history))
                     : "r"(&wrap(y, "in", &mut history)));
            }
            assert_eq!((x,y), (1,1));
            let b: &[_] = &["out", "in"];
            assert_eq!(history.as_slice(), b);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    fn out_write_only_expr_then_in_expr() {
        demo!("=r")
    }

    fn out_read_write_expr_then_in_expr() {
        demo!("+r")
    }

    out_write_only_expr_then_in_expr();
    out_read_write_expr_then_in_expr();
}

#[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
pub fn main() {}
