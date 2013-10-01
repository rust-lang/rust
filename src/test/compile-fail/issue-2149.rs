// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait vec_monad<A> {
    fn bind<B>(&self, f: &fn(A) -> ~[B]);
}

impl<A> vec_monad<A> for ~[A] {
    fn bind<B>(&self, f: &fn(A) -> ~[B]) {
        let mut r = fail2!();
        for elt in self.iter() { r = r + f(*elt); }
        //~^ WARNING unreachable expression
        //~^^ ERROR the type of this value must be known
   }
}
fn main() {
    ["hi"].bind(|x| [x] );
    //~^ ERROR type `[&'static str, .. 1]` does not implement any method in scope named `bind`
}
