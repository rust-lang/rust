// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct ClientMap;
pub struct ClientMap2;

pub trait Service {
    type Request;
    fn call(&self, _req: Self::Request);
}

pub struct S<T>(T);

impl Service for ClientMap {
    type Request = S<Box<Fn(i32)>>;
    fn call(&self, _req: Self::Request) {}
}


impl Service for ClientMap2 {
    type Request = (Box<Fn(i32)>,);
    fn call(&self, _req: Self::Request) {}
}


fn main() {
    ClientMap.call(S { 0: Box::new(|_msgid| ()) });
    ClientMap.call(S(Box::new(|_msgid| ())));
    ClientMap2.call((Box::new(|_msgid| ()),));
}
