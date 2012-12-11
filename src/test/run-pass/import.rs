// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    #[legacy_exports];
    fn x(y: int) { log(debug, y); }
}

mod bar {
    #[legacy_exports];
    use foo::x;
    use z = foo::x;
    fn thing() { x(10); z(10); }
}

fn main() { bar::thing(); }
