// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that autoderef of a type like this does not
// cause compiler to loop.  Note that no instances
// of such a type could ever be constructed.
enum t = @t; //~ ERROR this type cannot be instantiated

trait to_str_2 {
    fn to_str() -> ~str;
}

// I use an impl here because it will cause
// the compiler to attempt autoderef and then
// try to resolve the method.
impl t: to_str_2 {
    fn to_str() -> ~str { ~"t" }
}

fn new_t(x: t) {
    x.to_str();
}

fn main() {
}
