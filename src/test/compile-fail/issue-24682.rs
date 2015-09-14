// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A: Sized {
    type N;
    fn x() ->
        Self<
          N= //~ ERROR associated type bindings are not allowed here
          Self::N> {
        loop {}
    }
    fn y(&self) ->
        std
           <N=()> //~ ERROR associated type bindings are not allowed here
           ::option::Option<()>
    { None }
    fn z(&self) ->
        u32<N=()> //~ ERROR associated type bindings are not allowed here
    { 42 }

}

fn main() {
}
