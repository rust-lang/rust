// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type cat = {cat_name: ~str, cat_name: int};  //~ ERROR Duplicate field name cat_name

fn main()
{
  io::println(int::str({x: 1, x: 2}.x)); //~ ERROR Duplicate field name x
}
