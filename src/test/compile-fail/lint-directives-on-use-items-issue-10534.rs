// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_imports)]
#![allow(non_uppercase_statics)]

// The aim of this test is to ensure that deny/allow/warn directives
// are applied to individual "use" statements instead of silently
// ignored.

#[allow(dead_code)]
mod a { pub static x: int = 3; pub static y: int = 4; }

mod b {
    use a::x; //~ ERROR: unused import
    #[allow(unused_imports)]
    use a::y; // no error here
}

#[allow(unused_imports)]
mod c {
    use a::x;
    #[deny(unused_imports)]
    use a::y; //~ ERROR: unused import
}

fn main() {}
