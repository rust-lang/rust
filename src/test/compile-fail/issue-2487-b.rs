// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct socket {
    sock: int,
}

impl socket : Drop {
    fn finalize(&self) {}
}

impl socket {

    fn set_identity()  {
        do closure {
        setsockopt_bytes(self.sock) //~ ERROR copying a noncopyable value
      } 
    }
}

fn closure(f: fn@()) { f() }

fn setsockopt_bytes(+_sock: int) { }

fn main() {}
