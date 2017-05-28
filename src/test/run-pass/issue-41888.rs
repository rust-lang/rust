// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() { let _ = g(Some(E::F(K))); }

type R = Result<(), ()>;
struct K;

enum E {
    F(K), // must not be built-in type
    #[allow(dead_code)]
    G(Box<E>, Box<E>),
}

fn translate(x: R) -> R { x }

fn g(mut status: Option<E>) -> R {
    loop {
        match status {
            Some(infix_or_postfix) => match infix_or_postfix {
                E::F(_op) => { // <- must be captured by value
                    match Ok(()) {
                        Err(err) => return Err(err),
                        Ok(_) => {},
                    };
                }
                _ => (),
            },
            _ => match translate(Err(())) {
                Err(err) => return Err(err),
                Ok(_) => {},
            }
        }
        status = None;
    }
}
