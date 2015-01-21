// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test verifies that casting from the same lifetime on a value
// to the same lifetime on a trait succeeds. See issue #10766.

#![allow(dead_code)]
fn main() {
    trait T {}

    fn f<'a, V: T>(v: &'a V) -> &'a T {
        v as &'a T
    }
}
