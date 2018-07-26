// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: lexical nll
#![cfg_attr(nll, feature(nll))]

// Exercise the unused_mut attribute in some positive and negative cases

#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![deny(unused_mut)]


fn main() {
    // negative cases
    let mut a = 3; //[lexical]~ ERROR: variable does not need to be mutable
                   //[nll]~^ ERROR: variable does not need to be mutable
    let mut a = 2; //[lexical]~ ERROR: variable does not need to be mutable
                   //[nll]~^ ERROR: variable does not need to be mutable
    let mut b = 3; //[lexical]~ ERROR: variable does not need to be mutable
                   //[nll]~^ ERROR: variable does not need to be mutable
    let mut a = vec![3]; //[lexical]~ ERROR: variable does not need to be mutable
                         //[nll]~^ ERROR: variable does not need to be mutable
    let (mut a, b) = (1, 2); //[lexical]~ ERROR: variable does not need to be mutable
                             //[nll]~^ ERROR: variable does not need to be mutable
    let mut a; //[lexical]~ ERROR: variable does not need to be mutable
               //[nll]~^ ERROR: variable does not need to be mutable
    a = 3;

    let mut b; //[lexical]~ ERROR: variable does not need to be mutable
               //[nll]~^ ERROR: variable does not need to be mutable
    if true {
        b = 3;
    } else {
        b = 4;
    }

    match 30 {
        mut x => {} //[lexical]~ ERROR: variable does not need to be mutable
                    //[nll]~^ ERROR: variable does not need to be mutable
    }
    match (30, 2) {
      (mut x, 1) | //[lexical]~ ERROR: variable does not need to be mutable
                   //[nll]~^ ERROR: variable does not need to be mutable
      (mut x, 2) |
      (mut x, 3) => {
      }
      _ => {}
    }

    let x = |mut y: isize| 10; //[lexical]~ ERROR: variable does not need to be mutable
                               //[nll]~^ ERROR: variable does not need to be mutable
    fn what(mut foo: isize) {} //[lexical]~ ERROR: variable does not need to be mutable
                               //[nll]~^ ERROR: variable does not need to be mutable

    let mut a = &mut 5; //[lexical]~ ERROR: variable does not need to be mutable
                        //[nll]~^ ERROR: variable does not need to be mutable
    *a = 4;

    let mut a = 5;
    let mut b = (&mut a,); //[lexical]~ ERROR: variable does not need to be mutable
    *b.0 = 4;              //[nll]~^ ERROR: variable does not need to be mutable

    let mut x = &mut 1; //[lexical]~ ERROR: variable does not need to be mutable
                        //[nll]~^ ERROR: variable does not need to be mutable
    let mut f = || {
      *x += 1;
    };
    f();

    fn mut_ref_arg(mut arg : &mut [u8]) -> &mut [u8] {
        &mut arg[..] //[lexical]~^ ERROR: variable does not need to be mutable
                     //[nll]~^^ ERROR: variable does not need to be mutable
    }

    let mut v : &mut Vec<()> = &mut vec![]; //[lexical]~ ERROR: variable does not need to be mutable
                                            //[nll]~^ ERROR: variable does not need to be mutable
    v.push(());

    // positive cases
    let mut a = 2;
    a = 3;
    let mut a = Vec::new();
    a.push(3);
    let mut a = Vec::new();
    callback(|| {
        a.push(3);
    });
    let mut a = Vec::new();
    callback(|| {
        callback(|| {
            a.push(3);
        });
    });
    let (mut a, b) = (1, 2);
    a = 34;

    match 30 {
        mut x => {
            x = 21;
        }
    }

    match (30, 2) {
      (mut x, 1) |
      (mut x, 2) |
      (mut x, 3) => {
        x = 21
      }
      _ => {}
    }

    let x = |mut y: isize| y = 32;
    fn nothing(mut foo: isize) { foo = 37; }

    // leading underscore should avoid the warning, just like the
    // unused variable lint.
    let mut _allowed = 1;
}

fn callback<F>(f: F) where F: FnOnce() {}

// make sure the lint attribute can be turned off
#[allow(unused_mut)]
fn foo(mut a: isize) {
    let mut a = 3;
    let mut b = vec![2];
}

// make sure the lint attribute can be turned off on let statements
#[deny(unused_mut)]
fn bar() {
    #[allow(unused_mut)]
    let mut a = 3;
    let mut b = vec![2]; //[lexical]~ ERROR: variable does not need to be mutable
                         //[nll]~^ ERROR: variable does not need to be mutable
}
