// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #27592.

fn write<'a, F: ::std::ops::FnOnce()->::std::fmt::Arguments<'a> + 'a>(fcn: F) {
    use std::fmt::Write;
    let _ = match fcn() { a => write!(&mut Stream, "{}", a), };
}

struct Stream;
impl ::std::fmt::Write for Stream {
    fn write_str(&mut self, _s: &str) -> ::std::fmt::Result {
        Ok( () )
    }
}

fn main() {
    write(|| format_args!("{}", String::from("Hello world")));
    //~^ ERROR borrowed value does not live long enough
    //~| ERROR borrowed value does not live long enough
}
