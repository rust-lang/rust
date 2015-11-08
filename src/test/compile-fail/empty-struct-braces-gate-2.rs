// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Feature gate test for empty struct with braces
// Can't use braced expressions and patterns with structs defined without braces

struct Empty2;

enum E {
    Empty5,
}

fn main() {
    let e2: Empty2 = Empty2 {}; //~ ERROR empty structs and enum variants with braces are unstable
    let e2: Empty2 = Empty2;
    // Issue #28692
    // let e5: E = E::Empty5 {}; // ERROR empty structs and enum variants with braces are unstable
    let e5: E = E::Empty5;

    match e2 {
        Empty2 {} => {} //~ ERROR empty structs and enum variants with braces are unstable
    }
    match e2 {
        Empty2 => {}
    }
    match e2 {
        Empty2 { .. } => {} //~ ERROR empty structs and enum variants with braces are unstable
    }
    // Issue #28692
    // match e5 {
    //     E::Empty5 {} => {} // ERROR empty structs and enum variants with braces are unstable
    // }
    match e5 {
        E::Empty5 => {}
    }
    // Issue #28692
    // match e5 {
    //     E::Empty5 { .. } => {} // ERROR empty structs and enum variants with braces are unstable
    // }

    let e22 = Empty2 { ..e2 }; //~ ERROR empty structs and enum variants with braces are unstable
}
