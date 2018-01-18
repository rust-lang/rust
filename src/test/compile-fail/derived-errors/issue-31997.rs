// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the resolve failure does not lead to downstream type errors.
// See issue #31997.

trait TheTrait { }

fn closure<F, T>(x: F) -> Result<T, ()>
    where F: FnMut() -> T, T: TheTrait,
{
    unimplemented!()
}

fn foo() -> Result<(), ()> {
    try!(closure(|| bar(0 as *mut _))); //~ ERROR cannot find function `bar` in this scope
    Ok(())
}

fn main() { }
