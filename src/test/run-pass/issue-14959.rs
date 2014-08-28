// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(overloaded_calls)]

use std::ops::Fn;

trait Response {}
trait Request {}
trait Ingot<R, S> {
    fn enter(&mut self, _: &mut R, _: &mut S, a: &mut Alloy) -> Status;
}

#[allow(dead_code)]
struct HelloWorld;

struct SendFile<'a>;
struct Alloy;
enum Status {
    Continue
}

impl Alloy {
    fn find<T>(&self) -> Option<T> {
        None
    }
}

impl<'a, 'b> Fn<(&'b mut Response+'b,),()> for SendFile<'a> {
    extern "rust-call" fn call(&self, (_res,): (&'b mut Response+'b,)) {}
}

impl<Rq: Request, Rs: Response> Ingot<Rq, Rs> for HelloWorld {
    fn enter(&mut self, _req: &mut Rq, res: &mut Rs, alloy: &mut Alloy) -> Status {
        let send_file = alloy.find::<SendFile>().unwrap();
        send_file(res);
        Continue
    }
}

fn main() {}
