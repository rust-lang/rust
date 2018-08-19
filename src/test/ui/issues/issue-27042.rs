// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #27042. Test that a loop's label is included in its span.

fn main() {
    let _: i32 =
        'a: // in this case, the citation is just the `break`:
        loop { break }; //~ ERROR mismatched types
    let _: i32 =
        'b: //~ ERROR mismatched types
        while true { break }; // but here we cite the whole loop
    let _: i32 =
        'c: //~ ERROR mismatched types
        for _ in None { break }; // but here we cite the whole loop
    let _: i32 =
        'd: //~ ERROR mismatched types
        while let Some(_) = None { break };
}
