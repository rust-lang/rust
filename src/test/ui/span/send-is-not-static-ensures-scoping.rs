// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Guard<'a> {
    f: Box<Fn() + Send + 'a>,
}

fn scoped<'a, F: Fn() + Send + 'a>(f: F) -> Guard<'a> {
    Guard { f: Box::new(f) }
}

impl<'a> Guard<'a> {
    fn join(self) {}
}

fn main() {
    let bad = {
        let x = 1;
        let y = &x; //~ ERROR `x` does not live long enough

        scoped(|| {
            let _z = y;
            //~^ ERROR `y` does not live long enough
        })
    };

    bad.join();
}
