// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]
#![feature(slice_patterns)]
#![allow(unreachable_patterns)]
#![allow(unreachable_code)]

#[allow(dead_code)]
fn foo(z: !) {
    let x: Result<!, !> = Ok(z);

    let Ok(_y) = x;
    let Err(_y) = x;

    let x = [z; 1];

    match x {};
    match x {
        [q] => q,
    };
}

fn bar(nevers: &[!]) {
    match nevers {
        &[]  => (),
    };

    match nevers {
        &[]  => (),
        &[_]  => (),
        &[_, _, _, ..]  => (),
    };
}

fn main() {
    let x: Result<u32, !> = Ok(123);
    let Ok(y) = x;

    assert_eq!(123, y);

    match x {
        Ok(y) => y,
    };

    match x {
        Ok(y) => y,
        Err(e) => match e {},
    };

    let x: Result<u32, &!> = Ok(123);
    match x {
        Ok(y) => y,
    };

    bar(&[]);
}

