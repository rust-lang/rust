// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn process<T>(_t: T) {}

fn match_const_opt_by_mut_ref(v: &const Option<int>) {
    match *v {
      Some(ref mut i) => process(i), //~ ERROR cannot borrow
        //~^ ERROR unsafe borrow of aliasable, const value
      None => ()
    }
}

fn match_const_opt_by_const_ref(v: &const Option<int>) {
    match *v {
      Some(ref const i) => process(i),
        //~^ ERROR unsafe borrow of aliasable, const value
      None => ()
    }
}

fn match_const_opt_by_imm_ref(v: &const Option<int>) {
    match *v {
      Some(ref i) => process(i), //~ ERROR cannot borrow
        //~^ ERROR unsafe borrow of aliasable, const value
      None => ()
    }
}

fn match_const_opt_by_value(v: &const Option<int>) {
    match *v {
      Some(copy i) => process(i),
      None => ()
    }
}

fn main() {
}
