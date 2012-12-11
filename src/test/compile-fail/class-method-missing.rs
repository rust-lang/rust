// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:missing method `eat`
trait animal {
  fn eat();
}

struct cat {
  meows: uint,
}

impl cat : animal {
}

fn cat(in_x : uint) -> cat {
    cat {
        meows: in_x
    }
}

fn main() {
  let nyan = cat(0u);
}