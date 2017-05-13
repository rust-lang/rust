// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns)]
#![feature(slice_patterns)]
#![feature(box_syntax)]
#![feature(never_type)]
#![deny(unreachable_patterns)]

mod foo {
    pub struct SecretlyEmpty {
        _priv: !,
    }
}

struct NotSoSecretlyEmpty {
    _priv: !,
}

fn main() {
    let x: &[!] = &[];

    match x {
        &[]   => (),
        &[..] => (),    //~ ERROR unreachable pattern
    };

    let x: Result<Box<NotSoSecretlyEmpty>, &[Result<!, !>]> = Err(&[]);
    match x {
        Ok(box _) => (),    //~ ERROR unreachable pattern
        Err(&[]) => (),
        Err(&[..]) => (),   //~ ERROR unreachable pattern
    }

    let x: Result<foo::SecretlyEmpty, Result<NotSoSecretlyEmpty, u32>> = Err(Err(123));
    match x {
        Ok(_y) => (),
        Err(Err(_y)) => (),
        Err(Ok(_y)) => (),  //~ ERROR unreachable pattern
    }
}

