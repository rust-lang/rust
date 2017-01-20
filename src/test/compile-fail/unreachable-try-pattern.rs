// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]
#![deny(unreachable_code)]
#![deny(unreachable_patterns)]

enum Void {}

impl From<Void> for i32 {
    fn from(v: Void) -> i32 {
        match v {}
    }
}

fn bar(x: Result<!, i32>) -> Result<u32, i32> {
    x?
}

fn foo(x: Result<!, i32>) -> Result<u32, i32> {
    let y = (match x { Ok(n) => Ok(n as u32), Err(e) => Err(e) })?;
    //~^ ERROR unreachable pattern
    //~| ERROR unreachable expression
    Ok(y)
}

fn qux(x: Result<u32, Void>) -> Result<u32, i32> {
    Ok(x?)
}

fn vom(x: Result<u32, Void>) -> Result<u32, i32> {
    let y = (match x { Ok(n) => Ok(n), Err(e) => Err(e) })?;
    //~^ ERROR unreachable pattern
    Ok(y)
}

fn main() {
    let _ = bar(Err(123));
    let _ = foo(Err(123));
    let _ = qux(Ok(123));
    let _ = vom(Ok(123));
}

