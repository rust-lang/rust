// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn early<'a, 'b>(self) -> (&'a u8, &'b u8) { loop {} }
    fn life_and_type<'a, T>(&self) -> &'a T { loop {} }
}

fn main() {
    S.late(&0, &0); // OK
    S.late::<'static>(&0, &0);
    //~^ ERROR expected at most 0 lifetime parameters, found 1 lifetime parameter
    S.late::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR expected at most 0 lifetime parameters, found 3 lifetime parameter
    S.early(); // OK
    S.early::<'static>();
    //~^ ERROR expected 2 lifetime parameters, found 1 lifetime parameter
    S.early::<'static, 'static, 'static>();
    //~^ ERROR expected at most 2 lifetime parameters, found 3 lifetime parameters
    let _: &u8 = S.life_and_type::<'static>();
    S.life_and_type::<u8>();
    S.life_and_type::<'static, u8>();
}
