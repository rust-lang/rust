// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: copying a noncopyable value

struct my_resource {
  x: int,
}

impl my_resource : Drop {
    fn finalize(&self) {
        log(error, self.x);
    }
}

fn my_resource(x: int) -> my_resource {
    my_resource {
        x: x
    }
}

fn main() {
    {
        let a = {x: 0, y: my_resource(20)};
        let b = {x: 2,.. copy a};
        log(error, (a, b));
    }
}
