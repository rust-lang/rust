// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;

fn asSendfn( f : ~fn()->uint ) -> uint {
   return f();
}

fn asLambda( f : @fn()->uint ) -> uint {
   return f();
}

fn asBlock( f : &fn()->uint ) -> uint {
   return f();
}

pub fn main() {
   let x = asSendfn(|| 22u);
   assert!((x == 22u));
   let x = asLambda(|| 22u);
   assert!((x == 22u));
   let x = asBlock(|| 22u);
   assert!((x == 22u));
}
